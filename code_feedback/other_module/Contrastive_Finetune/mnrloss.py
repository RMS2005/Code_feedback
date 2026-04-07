# mnr_loss.py

import torch
from torch import nn, Tensor

class MultipleNegativesRankingLoss(nn.Module):
    """
    A robust implementation of Multiple Negatives Ranking (MNR) Loss for in-batch negatives.
    This loss is highly effective for training sentence embedding models.
    """
    def __init__(self, scale: float = 20.0):
        """
        Args:
            scale (float): The scaling factor for the cosine similarity, which is
                           equivalent to 1/temperature. A higher scale makes the
                           softmax distribution sharper, focusing on harder negatives.
                           A common value is 20.0.
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        # The CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss,
        # which is exactly what's needed for this type of ranking loss.
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor):
        """
        Calculates the MNR loss. This implementation assumes that for each anchor 
        in the batch, its "positive" counterpart is itself, and all other items in 
        the batch serve as "negative" examples.

        Args:
            embeddings (Tensor): A tensor of embeddings with shape 
                                 [batch_size, embedding_dim]. 
                                 It is crucial that these embeddings are L2-normalized 
                                 before being passed to this function.
        
        Returns:
            Tensor: The calculated MNR loss value for the batch.
        """
        # Ensure embeddings are on a device
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Calculate the cosine similarity between all pairs of embeddings in the batch.
        # For L2-normalized vectors, the dot product (matrix multiplication) is
        # equivalent to the cosine similarity.
        # Shape: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Scale the similarity scores. This helps in adjusting the focus of the loss.
        scaled_similarity_matrix = similarity_matrix * self.scale
        
        # The target labels for the cross-entropy loss are the indices of the
        # positive pairs. In this in-batch setup, the positive for sample `i`
        # (the anchor) is itself, located at index `i` in its similarity row.
        # Therefore, the labels are simply the diagonal indices: [0, 1, 2, ..., batch_size-1].
        target_labels = torch.arange(batch_size, device=device)
        
        # Calculate the cross-entropy loss.
        # This loss function internally applies a softmax to each row of the
        # `scaled_similarity_matrix` and then calculates the negative log-likelihood
        # of the correct class (the `target_labels`).
        # This effectively trains the model to maximize the similarity score at
        # `similarity_matrix[i, i]` for each `i`, relative to all other scores in that row.
        loss = self.cross_entropy_loss(scaled_similarity_matrix, target_labels)
        
        return loss