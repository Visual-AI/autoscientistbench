import torch

class ObservationLoader:
    def __init__(self, x, batch_size=400, device="cpu"):
        self.xdata = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.device = device
        self.dim = self.xdata.shape[1]

    def _process_batch(self, batch1, batch2):
        """
        Helper function to ensure Train and Test batches undergo 
        the EXACT same transformation and corruption logic.
        """
        # Randomly select one dimension to mask for each item in the batch
        indices = torch.randint(0, self.dim, (batch1.size(0),), device=self.device)
        mask = torch.zeros((batch1.size(0), self.dim), device=self.device)
        mask.scatter_(1, indices.unsqueeze(1), 1.0)
        
        # Create a batch index range on the correct device for indexing
        batch_idx = torch.arange(batch1.size(0), device=self.device)

        # Flip a coin for each item to potentially mask the next dimension
        # Only consider indices that are not the last dimension
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices < self.dim - 1)
        
        # Use device-aware indexing
        mask[batch_idx[should_mask_next], indices[should_mask_next] + 1] = 1.0
        
        # flip a coin for each item to potentially mask the previous dimension
        should_mask_prev = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices > 0)
        mask[batch_idx[should_mask_prev], indices[should_mask_prev] - 1] = 1.0
        
        return batch1, batch2, mask 

    def get_batch(self):
        """Returns a random batch."""
        batch1 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        batch2 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        return self._process_batch(batch1, batch2)

