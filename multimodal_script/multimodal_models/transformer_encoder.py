import torch
from torch import nn

class TransformerPatientPerEncoder(nn.Module):
    """
    Transformer Encoder Fusion Strategy.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime flags; may include `img_checkpoint` / `tab_checkpoint` to freeze encoders.
    img_model : nn.Module
        Image encoder module.
    tab_model : nn.Module
        Tabular encoder module.
    img_feature_dim : int
        Feature size produced by `img_model` before projection.
    tab_feature_dim : int
        Feature size produced by `tab_model` before projection.
    embed_dim : int
        Shared embedding size used after fusion (d_model for the encoder).
    num_heads : int
        Number of attention heads in each encoder layer.
    num_layers : int
        Number of layers in the Transformer encoder stack.
    num_classes : int, optional
        Number of output classes for the classifier head (default: 1).
    dropout : float, optional
        Dropout used inside the encoder layers (default: 0.3).
    class_dropout : float, optional
        Dropout used inside the classifier MLP (default: 0.1).
    """

    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim,
                 embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
        super().__init__()
        
        self.img_model = img_model   # Image encoder module
        self.tab_model = tab_model   # Tabular encoder module
        self.embed_dim = embed_dim 
        self.args = args
        
        # After projecting both branches to img_feature_dim, concatenation yields:
        combined_dim = img_feature_dim + img_feature_dim
        
        # Image branch projection to img_feature_dim 
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, img_feature_dim),
            nn.LayerNorm(img_feature_dim),
            nn.ReLU(inplace=True)
        )
                                      
        # Tabular branch projection to img_feature_dim (to match image side)
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feature_dim, img_feature_dim),
            nn.LayerNorm(img_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Map concatenated per-patient features to the model embedding dimension
        self.feature_embedding = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Transformer Encoder stack (expects (B, T, E) with batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final normalization before classification
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 4, num_classes)  # logits per patient
        )
        
    def forward(self, img_input, tab_input):
        """
        Forward pass:
          1) Extract features from image and tabular branches.
          2) Project, concatenate per patient, and encode with a Transformer.
          3) Normalize and classify.

        Parameters
        ----------
        img_input : Batch of images.
        tab_input : Batch of tabular data.

        Returns
        -------
        torch.Tensor
            Class logits tensor.
        """
        
        # ----- Image features -----
        # Extract features from the image model (frozen or trainable) ---
        with torch.no_grad() if self.args.img_checkpoint else torch.enable_grad():
            img_features = self.img_model(img_input)        
        
        # Project to img_feature_dim
        img_features = self.img_proj(img_features)          
        
        # ----- Tabular features -----
        # Extract features from the tabular model (frozen or trainable) ---
        with torch.no_grad() if self.args.tab_checkpoint else torch.enable_grad():
            tab_features = self.tab_model(tab_input)        
        
        # Project to img_feature_dim
        tab_features = self.tab_proj(tab_features)          
        
        # Concatenate along feature dimension
        combined_features = torch.cat((img_features, tab_features), dim=1)
        
        # Add a sequence dimension for the encoder
        combined_features = combined_features.unsqueeze(0)

        # Project to embed_dim
        x = self.feature_embedding(combined_features)

        # Encode with Transformer
        x = self.encoder(x)
        
        # Remove the added sequence dimension and normalize
        x = self.norm(x).squeeze(0)
        
        # Final classification
        logits = self.classifier(x)
        return logits
