import torch
from torch import nn
import sys
import pathlib 

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from tabular_script.tab_models.TabularTransformer import TabEncoder

class TabularModel:
    def __init__(self, args):
        """
        Initialize the ModelFactory with the given arguments.

        Args:
            args (Namespace): Parsed arguments containing model configuration.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        """
        Builds and initializes the model based on the provided arguments.

        Returns:
            nn.Module: The constructed model.
        """
        if self.args.tab_model == "TabTransformer":
            self.model = TabEncoder(num_features=self.args.num_features, dim_embedding=self.args.tab_feature_dim, num_heads=self.args.tab_num_heads, num_layers=self.args.tab_num_layers)     
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        return self.model
