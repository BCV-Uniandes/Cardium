import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    MLP Fusion Strategy
    Parameters
    ----------
    args : argparse.Namespace
        Must contain `n_classes` and flags like `img_checkpoint` and `tab_checkpoint`
        to decide whether to freeze the encoders during the forward pass.
    img_model : nn.Module
        Image encoder. Expected to output a vector per sample of size `img_feature_dim`.
    tab_model : nn.Module
        Tabular encoder. Expected to output a vector per sample of size `tab_feature_dim`.
    img_feature_dim : int
        Output dimension of `img_model` (and the target projection dimension).
    tab_feature_dim : int
        Output dimension of `tab_model` before projection.
    class_dropout : float, optional
        Dropout probability inside the classifier MLP.
    """

    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, class_dropout=0.1):
        super(MLP, self).__init__()
        self.img_model = img_model      # Image encoder module
        self.tab_model = tab_model      # Tabular encoder module
        self.args = args
        self.num_classes = args.n_classes

        # Dimension after concatenation
        combined_dim = img_feature_dim + img_feature_dim

        # Image branch projection to img_feature_dim 
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, img_feature_dim),
            nn.LayerNorm(img_feature_dim),
            nn.ReLU(inplace=True)
        )

        # Tabular branch projection from tab_feature_dim to img_feature_dim,
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feature_dim, img_feature_dim),
            nn.LayerNorm(img_feature_dim),
            nn.ReLU(inplace=True)
        )

        # Normalization after concatenation 
        self.norm = nn.LayerNorm(combined_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(combined_dim // 2, combined_dim // 4),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(combined_dim // 4, self.num_classes)  # Output logits
        )

    def forward(self, img_input, tab_input):
        """
        Forward pass:
          1) Extract features from image and tabular branches.
          2) Project both to the same dimension.
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

        # ----- Image branch -----
        # Extract features from the image model (frozen or trainable)
        with torch.no_grad() if self.args.frozen else torch.enable_grad():
            img_features = self.img_model(img_input)     
        
        # Project to img_feature_dim
        img_features = self.img_proj(img_features)           

        # ----- Tabular branch -----
        # Extract features from the image model (frozen or trainable)
        with torch.no_grad() if self.args.frozen else torch.enable_grad():
            tab_features = self.tab_model(tab_input)    
        
        # Project to img_feature_dim
        tab_features = self.tab_proj(tab_features)           

        # Fusion and normalization 
        combined_features = torch.cat((img_features, tab_features), dim=1)  
        combined_features = self.norm(combined_features)                   

        # Final classification
        output = self.classifier(combined_features)           
        return output