import torch
from torch import nn
import pathlib 
import sys 

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from multimodal_script.multimodal_models.mlp import MLP
from multimodal_script.multimodal_models.transformer_encoder import TransformerPatientPerEncoder
from multimodal_script.multimodal_models.transformer_decoder import TransformerDecoder
from multimodal_script.multimodal_models.transformer_self_cross import TransformerSelfCross
from multimodal_script.multimodal_models.transformer_cross_attention import TransformerCrossAttention
from multimodal_script.multimodal_models.transformer_cross_attention_unimodal import TransformerCrossAttentionUnimodal
from multimodal_script.multimodal_models.softmax import TransformerCrossAttentionSoftMax

class MultimodalModel:
    """
    Factory class to build and initialize multimodal models combining image and tabular data.

    This class supports several architectures including MLP, Transformer encoders, decoders,
    cross-attention mechanisms, and unimodal variants. The specific model is selected based
    on the `multimodal_model` argument provided in `args`.

    Attributes:
        img_model (nn.Module): Pretrained image encoder.
        tab_model (nn.Module): Pretrained tabular encoder.
        args (Namespace): Arguments containing model configuration and hyperparameters.
        device (torch.device): Device for computation (GPU if available, else CPU).
        model (nn.Module | None): The constructed model after calling `build_model`.
    """

    def __init__(self, img_model, tab_model, args):
        """
        Initialize the MultimodalModel factory.

        Args:
            img_model (nn.Module): Pretrained image encoder.
            tab_model (nn.Module): Pretrained tabular encoder.
            args (Namespace): Arguments containing model configuration and hyperparameters.
        """
        self.img_model = img_model
        self.tab_model = tab_model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        """
        Constructs the multimodal model according to the `args.multimodal_model` specification.

        Returns:
            nn.Module: The initialized model ready for training or inference.

        Raises:
            ValueError: If `args.multimodal_model` contains an unsupported model name.
        """
        # MLP Fusion strategy
        if self.args.multimodal_model == "mlp":
            self.model = MLP(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                class_dropout=self.args.class_dropout
            )

        # Transformer Encoder Fusion strategy
        elif self.args.multimodal_model == "TransEncoder":
            self.model = TransformerPatientPerEncoder(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                num_classes=self.args.n_classes,
                dropout=self.args.path_dropout,
                class_dropout=self.args.class_dropout
            )

        # Transformer Decoder Fusion strategy
        elif self.args.multimodal_model == "TransDecoder":
            self.model = TransformerDecoder(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                num_classes=self.args.n_classes,
                dropout=self.args.path_dropout,
                class_dropout=self.args.class_dropout
            )

        # Transformer Encoder with Cross-Attention Fusion strategy
        elif self.args.multimodal_model == "TransCross":
            self.model = TransformerSelfCross(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                num_classes=self.args.n_classes,
                dropout=self.args.path_dropout,
                class_dropout=self.args.class_dropout
            )

        # Double Transformer Decoder Fusion strategy (ours)
        elif self.args.multimodal_model == "TransDoubleCross":
            self.model = TransformerCrossAttention(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                num_classes=self.args.n_classes,
                dropout=self.args.path_dropout,
                class_dropout=self.args.class_dropout
            )

        # Unimodal variant (image or tabular only)
        elif self.args.multimodal_model == "unimodal":
            self.model = TransformerCrossAttentionUnimodal(
                args=self.args,
                img_model=self.img_model,
                tab_model=self.tab_model,
                img_feature_dim=self.args.img_feature_dim,
                tab_feature_dim=self.args.tab_feature_dim,
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                num_classes=self.args.n_classes,
                dropout=self.args.path_dropout,
                class_dropout=self.args.class_dropout
            )

        else:
            raise ValueError(f"Unsupported model: {self.args.multimodal_model}")

        return self.model
