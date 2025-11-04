import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from src.embedding_utils import generate_embeddings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoModel(nn.Module):
    """
    Wrapper for a pre-trained DINO model from Hugging Face.
    Uses the AutoImageProcessor for preprocessing and AutoModel for the model itself.
    """
    def __init__(self, model_name):
        super(DinoModel, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)       
        return outputs.pooler_output



class FineTunedHead(nn.Module):
    """
    A simple feedforward neural network head for fine-tuning embeddings.
    Consists of two linear layers with ReLU activation and dropout in between.
    The output embeddings are L2 normalized to ensure they lie on the unit hypersphere.

    Input:
    - `input_dim`: Dimension of the input embeddings.
    - `hidden_dim`: Dimension of the hidden layer.
    - `output_dim`: Dimension of the output embeddings.
    - `dropout`: Dropout rate for regularization.

    Methods:
    - `forward(...)`: Forward pass through the network.
    - `compute_sims(...)`: Compute average similarities at initial and final checkpoints.
    - `train_epoch(...)`: Train for one epoch.
    - `train_loop(...)`: Full training loop.
    """
    def __init__(self, input_dim: int, hidden_dim: int=512, output_dim:int=128, dropout=0.1):
        super(FineTunedHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()



    def forward(self, x):
        """
        Forward pass through the network.

        Input:
        - `x`: Input tensor of shape (batch_size, input_dim).

        Output:
        - Transformed tensor of shape (batch_size, output_dim), L2 normalized.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=1)
        return x
    


class FineTunedEmbedding(nn.Module):
    def __init__(self, base_model, head_model):
        super(FineTunedEmbedding, self).__init__()
        self.base_model = base_model
        self.base_model.eval()  # Freeze base model
        self.head_model = head_model
        self.embeddings = None
    
    def forward(self, x):
        with torch.no_grad():
            base_embedding = self.base_model(x)
        final_embedding = self.head_model(base_embedding)
        return final_embedding
    
    def get_embeddings(self, dataset, device):
        if self.embeddings is None:
            self.embeddings = generate_embeddings(self.base_model, dataset[:], device)
        return self.embeddings
    
    def load_precomputed_embeddings(self, emb_path):
        self.embeddings = torch.load(emb_path)
        return self.embeddings
    

if __name__ == "__main__":
    from src.dataset import ZarrDataset
    from torch.utils.data import DataLoader

    dino_tiny = DinoModel("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")

    dataset = ZarrDataset("pusht/pusht_cchi_v7_replay.zarr", nepisodes=1)
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    for batch in loader:
        print("Batch shape:", batch.shape)
        emb = generate_embeddings(dino_tiny, batch, DEVICE)
        print("Embeddings shape:", emb.shape)
        break

    model = FineTunedEmbedding(dino_tiny, FineTunedHead(input_dim=768, hidden_dim=512, output_dim=128)).to(DEVICE)
    model.load_precomputed_embeddings("embeddings/embeddings_dino_tiny.pth")