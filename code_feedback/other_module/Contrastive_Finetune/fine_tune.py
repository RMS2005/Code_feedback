import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import wandb
import json
import re

# --- Transformers, PEFT, and BitsAndBytes ---
# We are not using BitsAndBytes for quantization in this version to prioritize batch size and stability.
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM # Use the correct AutoModel class
from peft import get_peft_model, LoraConfig, TaskType

# --- Supervised Contrastive Loss ---
try:
    # Ensure you have the per-class MulSupConLoss in this file
    from mul_supcon_loss import MulSupConLoss 
except ImportError:
    print("ERROR: mul_supcon_loss.py not found. Please create it with the per-class MulSupConLoss implementation.")
    exit()

try:
    from mnrloss import MultipleNegativesRankingLoss # <-- IMPORT FROM NEW FILE
except ImportError:
    print("ERROR: mnr_loss.py not found. Please create it with the MultipleNegativesRankingLoss class.")
    exit()

# --- Configuration Dictionary ---
config = {

    "model_name": "nomic-ai/nomic-embed-code",
    "data_path": "all_codes.csv",
    "batch_size": 16, # CRITICAL: Larger batch size for effective contrastive learning
    "accumulation_steps": 2, # With a larger batch size, we can reduce accumulation
    "epochs": 15, # Contrastive learning can benefit from a few more epochs
    "learning_rate": 2e-6,
    "weight_decay": 0.01,
    "max_seq_length": 1024,
    "output_model_path": "./finetuned_nomic_combined_loss",
    "wandb_project_name": "autograder_embedding_finetuning",
    "wandb_run_name": f"nomic_combined_loss_bs16_lr2e-5",
    "temperature": 0.07, # For SupCon
    "mnr_scale": 20.0,   # For MNR Loss
    "loss_alpha": 0.5,   # Weight for MulSupCon
    "warmup_steps": 100
}

# --- 1. Custom Dataset Class ---
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_len = max_len
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        code_text = str(row['code_snippet'])
        label = row['multi_hot_labels']
        student_id = row['student_id']
        encoding = self.tokenizer.encode_plus(
            code_text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float), # Multi-hot labels are float
            'student_id': student_id
        }

# --- 2. Model Wrapper ---
class EmbeddingModelWithHead(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 256)
        )
        self._init_weights(self.projection_head)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        pooled_output = self._mean_pooling(outputs, attention_mask)
        # The head will run in mixed precision if autocast is used
        projected_output = self.projection_head(pooled_output)
        return torch.nn.functional.normalize(projected_output, p=2, dim=1, eps=1e-8)

# --- 3. Learning Rate Scheduler ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# --- 4. Main Training Function ---
def main():
    wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"], config=config)
    
    # --- Data Preparation (for Multi-Label) ---
    df = pd.read_csv(config["data_path"])
    df = df.dropna(subset=['code_snippet'])
    
    def extract_problem_id(student_id):
        match = re.search(r'q\d+', str(student_id), re.IGNORECASE)
        if match: return 'problem_' + match.group(0).lower()
        return 'problem_unknown'
    df['problem_id'] = df['student_id'].apply(extract_problem_id)
    
    def get_tier(score):
        try: score = float(score)
        except (ValueError, TypeError): return "tier_FAIL"
        if score > 80.0: return "tier_PASS"
        elif score > 0.0 and score <= 80.0: return "tier_PARTIAL"
        else: return "tier_FAIL"
    df['performance_tier'] = df['score_percentage'].apply(get_tier)
    
    # Each sample gets a list of two labels: its problem and its performance tier
    df['label_set'] = df.apply(lambda row: [row['problem_id'], row['performance_tier']], axis=1)

    mlb = MultiLabelBinarizer()
    multi_hot_labels = mlb.fit_transform(df['label_set'])
    df['multi_hot_labels'] = list(multi_hot_labels)

    print(f"Dataset size: {len(df)}")
    print(f"Number of unique classes: {len(mlb.classes_)}")
    print("Example classes:", mlb.classes_)
    
    # --- Model and Tokenizer Setup ---
    print("\n--- Loading Model in float16 (no quantization) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16, # Load directly in float16
        trust_remote_code=True,
        device_map="auto" # Let accelerate handle GPU placement
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id
    
    base_model.gradient_checkpointing_enable()

    # Use the correct target modules for the Qwen2-based model
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)
    model = EmbeddingModelWithHead(peft_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.projection_head.to(device) # Move the trainable head to the device
    
    print("\n--- PEFT Model Summary ---")
    peft_model.print_trainable_parameters()

    # --- DataLoader & Training Setup ---
    dataset = CodeDataset(df, tokenizer, max_len=config["max_seq_length"])
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(data_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=total_steps)
    
    criterion_supcon = MulSupConLoss(temperature=config["temperature"])
    criterion_mnr = MultipleNegativesRankingLoss(scale=config["mnr_scale"])
    
    scaler = torch.cuda.amp.GradScaler()

        # --- Training Loop with Combined Loss ---
    print(f"\n--- Starting LoRA Fine-Tuning with Combined Loss (Batch Size: {config['batch_size']}) ---")
    model.train()
    for epoch in range(config["epochs"]):
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for i, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                features = model(input_ids, attention_mask)
                normalized_features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-8)
                
                loss_supcon = criterion_supcon(normalized_features, labels)
                loss_mnr = criterion_mnr(normalized_features)
                
                alpha = config["loss_alpha"]
                combined_loss = (alpha * loss_supcon) + ((1 - alpha) * loss_mnr)

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(f"\nWarning: NaN/Inf loss detected at step {i}, skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            if wandb.run:
                wandb.log({
                    "total_loss": combined_loss.item(),
                    "supcon_loss": loss_supcon.item(),
                    "mnr_loss": loss_mnr.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            loop.set_postfix(loss=combined_loss.item())

    # --- Save Model ---
    print("\n--- Saving Model ---")
    model.base_model.save_pretrained(config["output_model_path"])
    tokenizer.save_pretrained(config["output_model_path"])
    torch.save(model.projection_head.state_dict(), f"{config['output_model_path']}/projection_head.pth")
    with open(f"{config['output_model_path']}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Fine-tuned artifacts saved to {config['output_model_path']}")
    wandb.finish()

if __name__ == "__main__":
    main()