import torch
import torch.nn as nn

class TransformerCrossAttentionUnimodal(nn.Module):
    """
    Unimodal CARDIUM model.

    Runs a single-modality decoding path selected by `args.unimodal` ("img" or "tab").
    Image and tabular features are projected to a shared `embed_dim`. The selected
    branch is decoded with a Transformer decoder whose target and memory come from
    the same modality; the result is normalized and mapped to logits via an MLP.

    Parameters
    ----------
    args : argparse.Namespace
        Should include flags like `frozen` and `unimodal` ("img" or "tab").
    img_model : nn.Module
        Image encoder module.
    tab_model : nn.Module
        Tabular encoder module.
    img_feature_dim : int
        Feature size produced by `img_model` before projection.
    tab_feature_dim : int
        Feature size produced by `tab_model` before projection.
    embed_dim : int
        Transformer/embedding dimension.
    num_heads : int
        Number of attention heads in the decoder.
    num_layers : int
        Number of decoder layers.
    num_classes : int, optional
        Number of output classes (default: 1).
    dropout : float, optional
        Dropout inside the decoder (default: 0.3).
    class_dropout : float, optional
        Dropout inside the classifier MLP (default: 0.1).
    """

    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim,
                 embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
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

        # Transformer Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
               
        # Two separate decoders for img and tab branches
        self.decoder_img = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_tab = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final normalization per branch before classification head
        self.final_norm_img = nn.LayerNorm(embed_dim)
        self.final_norm_tab = nn.LayerNorm(embed_dim)

        # Classifier heads
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
          2) Project both to the same dimension.
          3) Decode unimodally and classify.

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
        with torch.no_grad() if self.args.frozen else torch.enable_grad():
            img_features = self.img_model(img_input)
        
        # Project to embed_dim
        img_features = self.img_proj(img_features).unsqueeze(0)

        # ----- Tabular features -----
        # Extract features from the tabular model (frozen or trainable) ---
        with torch.no_grad() if self.args.frozen else torch.enable_grad():
            tab_features = self.tab_model(tab_input)
        
        # Project to embed_dim
        tab_features = self.tab_proj(tab_features).unsqueeze(0)
 
        # Decode selected modality and normalize
        if self.args.unimodal == "img":
            img_features = self.decoder_img(tgt=img_features, memory=img_features)
            output = self.final_norm_img(img_features).squeeze(0)
        elif self.args.unimodal == "tab":
            tab_features = self.decoder_tab(tgt=tab_features, memory=tab_features)
            output = self.final_norm_tab(tab_features).squeeze(0)
        else:
            print(f"{self.args.unimodal} is not a valid modality")

        # Final classification
        logits = self.classifier(output)
        return logits


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_weights = None   # to store self-attention weights
        self.cross_attn_weights = None  # to store cross-attention weights

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None):
        """
        """
        # --- Self-Attention: how target attends to itself ---
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True  # request attention weights
        )
        self.self_attn_weights = self_attn_weights  # cache for later visualization

        # --- Cross-Attention: how target attends to memory ---
        tgt3, cross_attn_weights = self.multihead_attn(
            tgt2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True
        )
        self.cross_attn_weights = cross_attn_weights  # cache for later visualization

        # Delegate to the base implementation (will re-run the official forward path)
        return super().forward(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal
        )
