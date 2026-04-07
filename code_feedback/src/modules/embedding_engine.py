import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os

# --- Constants for Fine-Tuned EmbeddingEngine ---
# This is now a path to a local directory, not a Hugging Face model ID.
FINETUNED_MODEL_PATH = "./other_module/contrastive_learning/finetuned_nomic_mulsupcon_bs16_2" # UPDATE THIS to your final model path
# This should be the path to the saved projection head state dictionary.
PROJECTION_HEAD_PATH = "./other_module/contrastive_learning/finetuned_nomic_mulsupcon_bs16_2/projection_head.pth" # UPDATE THIS

MAX_SEQ_LENGTH = 1024 # Must match the sequence length used during training

# --- Re-define the Model Wrapper Class ---
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
        # The projection head will be on the same device, so this is safe
        projected_output = self.projection_head(pooled_output)
        return torch.nn.functional.normalize(projected_output, p=2, dim=1, eps=1e-8)


class EmbeddingEngine:
    def __init__(self):
        # --- DEFINE THE DEVICE EXPLICITLY AT THE TOP ---
        # Force the use of 'cuda:0' if available, otherwise fallback to 'cpu'.
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        print(f"[EMBEDDING_ENGINE] Initializing on explicitly set device: {self.device}")

        self.tokenizer = None
        self.model = None
        
        self._load_finetuned_model()

    def _load_finetuned_model(self):
        """
        Loads the fine-tuned model and projection head onto a single, specified GPU.
        """
        print(f"[EMBEDDING_ENGINE] Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")

        if not os.path.isdir(FINETUNED_MODEL_PATH) or not os.path.exists(PROJECTION_HEAD_PATH):
            print(f"[EMBEDDING_ENGINE] WARNING: Fine-tuned model directory '{FINETUNED_MODEL_PATH}' not found. Falling back to a standard pre-trained model for similarity check.")
            self._load_fallback_model()
            return

        try:
            # 1. Load the base model, but do NOT use device_map="auto".
            #    We will manually move it to our target device.
            base_model = AutoModelForCausalLM.from_pretrained(
                FINETUNED_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                # device_map="auto" removed
            )
            self.tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
            
            # 2. Create the wrapper
            self.model = EmbeddingModelWithHead(base_model)
            
            # 3. Load the projection head's state dict, mapping it to our target device
            print(f"  [EMBEDDING_ENGINE] Loading projection head state to: {self.device}")
            projection_head_state_dict = torch.load(PROJECTION_HEAD_PATH, map_location=self.device)
            self.model.projection_head.load_state_dict(projection_head_state_dict)
            
            # 4. Move the entire composite model to our target device.
            print(f"  [EMBEDDING_ENGINE] Moving entire model to: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[EMBEDDING_ENGINE] Fine-tuned model and projection head loaded successfully onto {self.device}.")

        except Exception as e:
            print(f"[EMBEDDING_ENGINE] ERROR: Could not load fine-tuned model.")
            print(f"[EMBEDDING_ENGINE] Details: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Loads a generic embedding model using sentence-transformers as a robust fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[EMBEDDING_ENGINE] Initializing fallback model: all-MiniLM-L6-v2...")
            self.fallback_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print(f"[EMBEDDING_ENGINE] Fallback model loaded successfully.")
        except Exception as e:
            print(f"[EMBEDDING_ENGINE] ERROR: Could not load fallback model: {e}")
            self.fallback_model = None

    def get_code_embedding(self, code_snippet: str) -> list | None:
        """
        Generates a vector embedding for a given code snippet.
        """
        if not self.model or not self.tokenizer:
            print("  [EMBEDDING_ENGINE] Fine-tuned model not available.")
            return None
        
        try:
            # The model and tokenizer are already on the correct device.
            inputs = self.tokenizer(
                [code_snippet],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=True
            )
            
            # Move the input tensors to the same device as the model.
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embedding_tensor = self.model(**inputs)
            
            embedding_list = embedding_tensor.squeeze().cpu().numpy().tolist()
            return embedding_list

        except Exception as e:
            print(f"  [EMBEDDING_ENGINE] ERROR during fine-tuned embedding generation: {e}")
        
        # --- Fallback Logic ---
        if hasattr(self, 'fallback_model') and self.fallback_model:
            try:
                embedding = self.fallback_model.encode(code_snippet).tolist()
                return embedding
            except Exception as e_fallback:
                print(f"  [EMBEDDING_ENGINE] Fallback also failed: {e_fallback}")

        return None

    def analyze(self, submission: dict) -> dict:
        # This method remains the same as your last working version.
        student_id = submission['student_id']
        full_code = submission.get('code', '')
        print(f"  [EMBEDDING_ENGINE] Generating fine-tuned embedding for {student_id}...")

        if 'embedding' not in submission['analysis']:
            submission['analysis']['embedding'] = {}
        
        submission['analysis']['embedding'].update({
            'model': FINETUNED_MODEL_PATH,
            'code_embedding': None
        })

        if self.model and self.tokenizer and full_code:
            embedding = self.get_code_embedding(full_code)
            if embedding:
                submission['analysis']['embedding']['code_embedding'] = embedding
                print(f"    [EMBEDDING_ENGINE] Fine-tuned embedding generated.")
        elif not full_code:
            print(f"    [EMBEDDING_ENGINE] No code provided to generate embedding.")
        
        return submission