import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerCrossAttention(nn.Module):
    """
    CARDIUM model.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime flags; should include `frozen` to optionally run encoders under no_grad.
    img_model : nn.Module
        Image encoder module.
    tab_model : nn.Module
        Tabular encoder module.
    img_feature_dim : int
        Feature size produced by `img_model` before projection.
    tab_feature_dim : int
        Feature size produced by `tab_model` before projection.
    embed_dim : int
        Shared embedding size used by projections and decoders (d_model).
    num_heads : int
        Number of attention heads in each decoder layer.
    num_layers : int
        Number of layers in each Transformer decoder stack.
    num_classes : int, optional
        Number of output classes for the classifier head (default: 1).
    dropout : float, optional
        Dropout used inside the decoder layers (default: 0.3).
    class_dropout : float, optional
        Dropout used inside the classifier MLP (default: 0.1).
    """
    
    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
        super().__init__()
        
        self.img_model = img_model  # Image model
        self.tab_model = tab_model  # Tabular model
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

        # Transformer Decoder
        decoder_layer = CustomTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        # Transformer decoder layers for each modality
        self.decoder_img = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_tab = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final normalization (per branch) before classification
        self.final_norm_img = nn.LayerNorm(embed_dim)
        self.final_norm_tab = nn.LayerNorm(embed_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim // 2),
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
          2) Project and perform cross-modal decoding.
          3) Concatenate, normalize, and classify.

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
        
        # --- Cross-modal decoding ---
        # Image decoder: queries are image features; keys/values are tab features
        img_features = self.decoder_img(
            tgt=img_features,  # queries (images)
            memory=tab_features  # keys/values (tabular)
        )  
        
        # Tab decoder: queries are tabular features; keys/values are image features
        tab_features = self.decoder_tab(
            tgt=tab_features,  # queries (tabular)
            memory=img_features  # keys/values (images)
        )  

        ## Normalize each branch, then fuse by concatenation
        output_img = self.final_norm_img(img_features).squeeze(0)
        output_tab = self.final_norm_tab(tab_features).squeeze(0)
        output = torch.cat((output_img, output_tab), dim=1)
        
        # Final classification
        logits = self.classifier(output)

        return logits

# Modified Decoder to store attention weigths 
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables to store attention weights
        self.self_attn_weights = None       
        self.cross_attn_weights = None     

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None):
        """
        Modified forward pass that stores attention weights.
        Parameters:
        - tgt: query sequence from the same modality (e.g., image or tabular features)
        - memory: key-value sequence from another modality
        - tgt_mask: attention mask for the query sequence
        - memory_mask: attention mask for memory (cross-attention)
        - tgt_key_padding_mask: mask to ignore padding positions in tgt
        - memory_key_padding_mask: mask to ignore padding positions in memory
        - tgt_is_causal / memory_is_causal: for causal attention (optional)
        """

        # ----------------------------
        # SELF-ATTENTION
        # ----------------------------
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,               # query, key, value = tgt
            attn_mask=tgt_mask,          # optional mask
            key_padding_mask=tgt_key_padding_mask,  # optional padding mask
            need_weights=True            # return attention weights
        )
        # Store the self-attention weights
        self.self_attn_weights = self_attn_weights

        # ----------------------------
        # CROSS-ATTENTION
        # ----------------------------
        tgt3, cross_attn_weights = self.multihead_attn(
            tgt2, memory, memory,        # query=tgt2, key=memory, value=memory
            attn_mask=memory_mask,       # optional mask
            key_padding_mask=memory_key_padding_mask,  # optional padding mask
            need_weights=True            # return attention weights
        )
        # Store the cross-attention weights
        self.cross_attn_weights = cross_attn_weights

        # ----------------------------
        # CALL THE ORIGINAL FORWARD
        # ----------------------------
        return super().forward(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal
        )