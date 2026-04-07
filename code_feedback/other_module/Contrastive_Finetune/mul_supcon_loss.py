# supcon_loss.py (Final, Corrected version that matches the paper)
import torch
import torch.nn as nn

class MulSupConLoss(nn.Module):
    """
    Implementation of the Multi-Label Supervised Contrastive Loss (MulSupCon)
    from the paper: https://arxiv.org/abs/2402.01613
    """
    def __init__(self, temperature=0.07):
        super(MulSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): Normalized embeddings [batch_size, embedding_dim]
            labels (torch.Tensor): Multi-hot encoded labels [batch_size, num_classes]
        """
        device = features.device
        batch_size = features.shape[0]

        # --- Step 1: Compute Similarity & Log-Probabilities ---
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # --- Step 2: Multi-Label Loss Calculation (Per-Class) ---
        # `positive_mask`[i, j, k] = 1 if sample i and j both have label k.
        # Shape: [batch_size, batch_size, num_classes]
        positive_mask = labels.unsqueeze(1) * labels.unsqueeze(0)
        positive_mask.diagonal(dim1=0, dim2=1).fill_(0) # Exclude self-comparisons
        
        # Number of positive pairs for each anchor, for each class
        # Shape: [batch_size, num_classes]
        num_positives_per_class = positive_mask.sum(1)
        
        # Sum of log-probabilities for positive pairs, per anchor, per class
        # log_prob is [batch_size, batch_size]. Expand to [batch_size, batch_size, 1] for broadcasting
        log_prob_expanded = log_prob.unsqueeze(2)
        sum_log_prob_pos = (positive_mask * log_prob_expanded).sum(1)
        
        # --- Loss Calculation ---
        # Denominator should not be zero. Where num_positives is 0, loss is 0.
        # Create a mask to avoid division by zero
        has_positives_mask = num_positives_per_class > 0
        
        # Initialize loss for each class to zero
        loss_per_class = torch.zeros_like(sum_log_prob_pos)
        
        # Calculate loss only where there are positive pairs
        loss_per_class[has_positives_mask] = -sum_log_prob_pos[has_positives_mask] / num_positives_per_class[has_positives_mask]
        
        # Final loss for an anchor is the sum of losses over all labels it has.
        # `labels` here acts as a mask.
        loss_per_anchor = (loss_per_class * labels).sum(1)
        
        # Final batch loss is the mean over anchors that had at least one label.
        # This prevents anchors with no labels (if that's possible in your data) from affecting the mean.
        num_labels_per_anchor = labels.sum(1)
        # Avoid division by zero if an anchor has no labels
        valid_anchors_mask = num_labels_per_anchor > 0
        if not torch.any(valid_anchors_mask):
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = loss_per_anchor[valid_anchors_mask].mean()

        return loss