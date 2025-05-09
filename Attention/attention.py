import torch
import torch.nn.functional as F

class SelfAttention:
    """
    SelfAttention is a class that implements a single-head self-attention mechanism, 
    commonly used in transformer-based models for sequence processing tasks.
    Attributes:
        emb_dim (int): The dimensionality of the embedding space for the attention mechanism.
        init_params (bool): A flag indicating whether the weight matrices have been initialized.
    Methods:
        __init__(emb_dim):
            Initializes the SelfAttention class with the specified embedding dimension.
        _init_params(in_features: int):
            Initializes the weight matrices (W_q, W_k, W_v) used for query, key, and value projections.
            This method is called internally during the first forward pass.
        forward(x):
            Computes the self-attention output for the input tensor `x`.
            The method performs the following steps:
            1. Initializes the weight matrices if not already initialized.
            2. Computes the query (Q), key (K), and value (V) projections.
            3. Calculates attention scores using scaled dot-product attention.
            4. Applies the softmax function to obtain attention weights.
            5. Computes the weighted sum of values (V) using the attention weights.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim) after applying self-attention.
        __call__(x):
            Allows the class instance to be called as a function, invoking the `forward` method.
            This is a convenience method for compatibility with PyTorch-style modules.
    """
    
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.init_params = False

    def _init_params(self, in_features: int):
        """ Initialize Self Attention's parameters"""
        
        self.W_q = torch.randn(in_features, self.emb_dim)
        self.W_k = torch.randn(in_features, self.emb_dim)
        self.W_v = torch.randn(in_features, self.emb_dim)

    def forward(self, x):
        """Forward propagation for Self-Attention"""
        
        # Check if parameters have already been initialized for this object
        # If not, we will initialize parameters for this object, knowing the shape of the inputs
        if not self.init_params:
            in_features = x.shape[-1]
            self._init_params(in_features)
            self.init_params = True

        # Get Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # calculate attention weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        outputs = torch.matmul(attn_weights, V)

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MultiHeadAttention:
    """
    Implements the Multi-Head Attention mechanism, a key component in Transformer models.
    Multi-Head Attention allows the model to jointly attend to information from different 
    representation subspaces at different positions. This implementation splits the input 
    embedding into multiple heads, applies self-attention to each head independently, and 
    then concatenates the results before applying a final linear transformation.
    Attributes:
        emb_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head, calculated as emb_dim // num_heads.
        heads (list): A list of `SelfAttention` instances, one for each attention head.
        W_0 (torch.Tensor): A learnable weight matrix for the final linear transformation.
    Methods:
        forward(x):
            Computes the multi-head attention output for the given input tensor.
        __call__(x):
            A convenience method to call the `forward` method directly.
    Raises:
        ValueError: If the embedding dimension (emb_dim) is not divisible by the number of heads (num_heads).
    Example:
        -> mha = MultiHeadAttention(emb_dim=128, num_heads=8)
        -> x = torch.randn(32, 10, 128)  # (batch_size, seq_len, emb_dim)
        -> output = mha(x)
        -> print(output.shape)  # Output shape: (32, 10, 128)
        """

    def __init__(self, emb_dim, num_heads):
        if emb_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        self.W_0 = torch.randn(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_transposed = x_reshaped.permute(0, 2, 1, 3)  # (batch, head, seq_len, head_dim)

        output_heads = []
        for i, head in enumerate(self.heads):
            head_input = x_transposed[:, i, :, :]
            head_output = head(head_input)
            output_heads.append(head_output)

        concatenated = torch.cat(output_heads, dim=-1)  # (batch, seq_len, emb_dim)
        outputs = concatenated @ self.W_0

        return outputs

    def __call__(self, x):
        return self.forward(x)


def test_attention_modules():
    torch.manual_seed(42)

    batch_size = 16
    seq_len = 65
    emb_dim = 512
    num_heads = 8

    print("Testing Self-Attention...")

    x_self = torch.randn(batch_size, seq_len, emb_dim)
    self_attn = SelfAttention(emb_dim)
    out_self = self_attn(x_self)

    assert out_self.shape == (batch_size, seq_len, emb_dim), \
        f"Expected shape {(batch_size, seq_len, emb_dim)}, got {out_self.shape}"
    print("✅ Self-Attention passed. Output shape:", out_self.shape)

    print("\nTesting MultiHeadAttention...")

    x_multi = torch.randn(batch_size, seq_len, emb_dim)
    multi_attn = MultiHeadAttention(emb_dim, num_heads)
    out_multi = multi_attn(x_multi)

    assert out_multi.shape == (batch_size, seq_len, emb_dim), \
        f"Expected shape {(batch_size, seq_len, emb_dim)}, got {out_multi.shape}"
    print("✅ MultiHeadAttention passed. Output shape:", out_multi.shape)


if __name__ == "__main__":
    test_attention_modules()