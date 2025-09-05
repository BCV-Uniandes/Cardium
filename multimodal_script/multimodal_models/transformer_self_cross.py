import torch
import torch.nn as nn

class TransformerSelfCross(nn.Module):
    """
    Self-then-cross attention multimodal classifier.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime flags; may include `img_checkpoint` / `tab_checkpoint` to freeze encoders.
    img_model : nn.Module
        Image encoder module.
    tab_model : nn.Module
        Tabular encoder module.
    img_feature_dim : int
        Feature size from the image encoder before projection.
    tab_feature_dim : int
        Feature size from the tabular encoder before projection.
    embed_dim : int
        Shared embedding size used after projection and inside the transformers.
    num_heads : int
        Number of attention heads in each transformer layer.
    num_layers : int
        Number of layers in each transformer stack and in the cross-attention stack.
    num_classes : int, optional
        Number of output classes for the classifier head (default: 1).
    dropout : float, optional
        Dropout used inside encoder and fusion layers (default: 0.3).
    class_dropout : float, optional
        Dropout used inside the classifier MLP (default: 0.1).
    """

    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
        super().__init__()
        
        self.img_model = img_model # Image encoder module
        self.tab_model = tab_model # Tabular encoder module
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
        
        # Modality-specific Transformer encoders (self-attention per modality)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tab_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.img_norm = nn.LayerNorm(embed_dim)
        self.tab_norm = nn.LayerNorm(embed_dim)

        # Cross-attention fusion stack (image queries attend to tabular keys/values)
        self.cross_attention_stack = nn.ModuleList([
            CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization before classification
        self.final_norm = nn.LayerNorm(embed_dim)

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
          2) Project, encode per modality, and fuse via stacked cross-attention.
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
        # Extract features from the image model
        with torch.no_grad():
            img_features = self.img_model(img_input)
        
        # Project to embed_dim
        img_features = self.img_proj(img_features).unsqueeze(0)

        # ----- Tabular features -----
        # Extract features from the tabular model
        with torch.no_grad():
            tab_features = self.tab_model(tab_input)
        
        # Project to embed_dim
        tab_features = self.tab_proj(tab_features).unsqueeze(0)
        
        # Self-attention encoders per modality
        img_features = self.img_encoder(img_features)
        tab_features = self.tab_encoder(tab_features)
        
        # Normalize encoder outputs
        img_features = self.img_norm(img_features)
        tab_features = self.tab_norm(tab_features)

        # Stacked cross-attention (image queries over tabular keys/values)
        for cross_attention_layer in self.cross_attention_stack:
            fused_features = cross_attention_layer(query=img_features, key_value=tab_features)
            
        # Remove the added sequence dimension and normalizes
        output = self.final_norm(fused_features).squeeze(0)
        
        # Final classification
        logits = self.classifier(output)
        
        return logits


class CrossAttentionFusion(nn.Module):
    """
    Single cross-attention block with residual, dropout, and LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))


# Cross-Attention Layer
class CrossAttentionFusion(nn.Module):
    """
    Single cross-attention block:
      query attends to key/value (with residual, dropout, and LayerNorm).
    Uses `nn.MultiheadAttention` with `batch_first=True` for (B, T, E) inputs.

    Inputs
    ------
    query     : Tensor  (B, Tq, E)
    key_value : Tensor  (B, Tk, E)

    Output
    ------
    Tensor : (B, Tq, E)   # normalized residual output
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        # Cross-attention: queries attend to keys/values from the other stream
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        # Residual connection + dropout + LayerNorm
        return self.norm(query + self.dropout(attn_output))