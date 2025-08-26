import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    """
    Transformer-Decoder Fusion Strategy.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime flags; may include `img_checkpoint` / `tab_checkpoint` to freeze encoders.
    img_model : nn.Module
        Image feature extractor.
    tab_model : nn.Module
        Tabular feature extractor.
    img_feature_dim : int
        Feature size produced by `img_model` before projection.
    tab_feature_dim : int
        Feature size produced by `tab_model` before projection.
    embed_dim : int
        Shared embedding size used by projections and decoder (d_model).
    num_heads : int
        Number of attention heads in each decoder layer.
    num_layers : int
        Number of layers in the Transformer decoder stack.
    num_classes : int, optional
        Number of output classes for the classifier head (default: 1).
    dropout : float, optional
        Dropout used inside the decoder layers (default: 0.3).
    class_dropout : float, optional
        Dropout used inside the classifier MLP (default: 0.1).
    """

    def __init__(
        self,
        args,
        img_model,
        tab_model,
        img_feature_dim,
        tab_feature_dim,
        embed_dim,
        num_heads,
        num_layers,
        num_classes: int = 1,
        dropout: float = 0.3,
        class_dropout: float = 0.1,
    ):
        super().__init__()

        self.img_model = img_model   # Image encoder module
        self.tab_model = tab_model   # Tabular encoder module
        self.embed_dim = embed_dim
        self.args = args

        # Image branch projection to embed_dim
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Tabular branch projection to embed_dim
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Transformer Decoder (cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Normalization after decoding and before classification
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, img_input, tab_input):
        """
        Forward pass:
          1) Extract features from image and tabular branches.
          2) Project and decode with cross-attention.
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
        # Extract image features (freeze dynamically if args.img_checkpoint is truthy) ---
        with torch.no_grad() if self.args.img_checkpoint else torch.enable_grad():
            img_features = self.img_model(img_input)

        # Project to embed_dim
        img_features = self.img_proj(img_features).unsqueeze(0)

        # ----- Tabular features -----
        # Extract tabular features (freeze dynamically if args.tab_checkpoint is truthy) ---
        with torch.no_grad() if self.args.tab_checkpoint else torch.enable_grad():
            tab_features = self.tab_model(tab_input)

        # Project to embed_dim 
        tab_features = self.tab_proj(tab_features).unsqueeze(0)

        # Cross-attention decoding
        decoded_features = self.decoder(
            tgt=img_features,
            memory=tab_features
        )

        # Remove the added sequence dimension and normalize
        decoded_features = self.norm(decoded_features).squeeze(0)

        # Final classification
        logits = self.classifier(decoded_features)
        return logits
