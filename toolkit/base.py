from torch import nn

class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        output_size (int): The number of output features.
        bias (boolean): Add bias to the linear layer
        layer_norm (boolean): Apply layer normalization
        dropout (float): The dropout value
        activation (nn.Module): The activation function to be applied after each fully connected layer.

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU())

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x