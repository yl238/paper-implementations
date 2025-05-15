import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class MultiHeadAttention(nn.Module):
    """The MultiHeadAttention class encapsulates the multi-head attention
    mechanism commonly used in transfomer models. It takes care of
    splitting the input into multiple attention heads, applying attention
    to each head, and then combining the results.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by
        # the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = (
            d_model // num_heads
        )  # Dimension of each head's key, query and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores. Here, the attention scores are
        # calculated by taking the dot product of queries (Q) and keys (K), and
        # then scaling by the square root of the key dimension (d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply mask if provided (useful for preventing attention to
        # certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention.
        # It enables the model to process multiple attention heads concurrently,
        # allowing for parallel computation.
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(
            1, 2
        )

    def combine_heads(self, x):
        # After applying attention to each head separately, this method
        # combines the multiple heads back to original shape of
        # (batch_size, seq_length, d_model).
        batch_size, _, seq_length, d_k = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        )

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """PositionWiseFeedForward class defines a position-wise
    feed-forward NN that consists of two linear layers with a ReLU
    activation function in between. In the context of transformer models,
    this feed-forward network is applied to each position separately
    and identically. It helps in transforming the features learned by the
    attention mechanism within the transformer, acting as an
    additional processing step for the attention outputs.
    """

    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """The PositionalEncoding class adds information about the
    position of tokens within the sequence. Since the transformer
    model lacks inherent knowledge of the order of tokens (due to
    its self-attention mechanism), this class helps the model to
    consider the position of tokens in the sequence. The sinusoidal functions
    used are chosen to allow the model to easily learn to attend
    to relative positions, as they produce a unique and smooth encoding
    for each position in the sequence.
    """

    def __init__(self, d_model, max_seq_length):
        """Initialization

        Args:
            d_model (int): The dimension of the model's input
            max_seq_length (int): The maximum length of the sequence
                for which positional encodings are pre-computed.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        # Position indices for each position in the sequence
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Scale the position indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # Apply sine function to the even indices and the cosine function to the odd
        # indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register pe as a buffer, which means it will be part of the module's
        # state but will not be considered a trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # The forward method simply adds the positional encoding to the
        # input x. It uses the first `x.size(1)` elements of `pe` to ensure that
        # the positional encoding matches the actual sequence length of `x`.
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    """The EncoderLayer class defines a single layer of the transformer's
    encoder. It encapsulates a mult-head self-attention mechanism followed
    by the position-wise feed-forward neural network, with residual
    connections, layer normalisation, and dropout applied. Together, these
    components allow the encoder to capture complex relationships in the
    input data and transform them into a useful representation for
    downstream tasks. Typically, multiple such encoder layers are stacked
    to form the complete encoder part of a transformer model.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """Initialization

        Args:
            d_model (int): The dimensionality of the input
            num_heads (int): The number of attention heads in the multi-head
                attention
            d_ff (int): The dimensionality of the inner layer in the
                position-wise feed-forward network
            dropout (float): The dropout rate used for regularization
        """
        super(EncoderLayer, self).__init__()
        # Multi-head attention mechanism
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Position-wise feed-forward NN
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Layer normalisation applied to smooth the layers' input
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout layer, used to prevent overfitting by randomly
        # setting some activations to zero during training.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Input x is passed through the multi-head attention mechanism
        attn_output = self.self_attn(x, x, x, mask)
        # Add and normalise after attention
        x = self.norm1(x + self.dropout(attn_output))
        # Feed forward network followed by dropout and norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """The DecoderLayer class defines a single layer of the transformer's
    decoder. It consists of a multi-head self-attention mechanism, a
    multi-head cross-attention mechanism (that attends to the encoder's
    output), a position-wise feed-forward neural network, and the
    corresponding residual connections, layer normalization, and dropout layers.
    This combination enables the decoder to generate meaningful outputs
    based on the encoder's representation, taking into account both the
    target sequence and the source sequence. As with the encoder,
    multiple decoder layers are typically stacked to form the complete
    decoder part of a transformer model.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # Multi-head self-attention mechanism for the target sequence.
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Multi-head attention mechanism that attends to the
        # encoder's output
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention on target sequence. The input x is processed
        # through a self-attention mechanism.
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Add and normalize (after self-attention)
        x = self.norm1(x + self.dropout(attn_output))
        # Cross-attention with encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        # Add and normalize (after cross-attention)
        x = self.norm2(x + self.dropout(attn_output))
        # Feed-forward network
        ff_output = self.feed_forward(x)
        # Add and normalize
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        """Constructor

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            d_model (int): The dimensionality of the model's embeddings.
            num_heads (int): Number of attention heads in the multi-head attention
                mechanism.
            num_layers (int): Number of layers for both the encoder and the decoder
            d_ff (int): Dimensionality of the inner layer in the feed-forward network.
            max_seq_length (int): Maximum sequence length for positional encoding.
            dropout (float): Dropout rate for regularization.
        """
        super(Transformer, self).__init__()
        # Embedding layer for the source sequence
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # Embedding layer for the target sequence
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding component.
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # A list of encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        # A list of decoder layers.
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        # Final fully connected (linear) layer mapping to target vocabulary size.
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # Dropout layer.
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """This method is used to create masks for the source and target
        sequences, ensuring that padding tokens are ignored and that future
        tokens are not visible during training for the target sequence.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """Forward pass for the Transformer"""
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # Input embedding and positional encoding: The source and target sequence are
        # first embedded using their respective embedding layers and then added to
        # their positional encodings
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        # Encoder layers: The source sequence is passed through the encoder layers,
        # with the final encoder output representing the processed source sequence.
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Decoder layers: The target sequence and the encoder's output are passed
        # through the decoder layers, resulting in the decoder's output.
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Final linear layer: The decoder's output is mapped to the target vocabulary
        # size using a fully connected (linear) layer
        output = self.fc(dec_output)
        return output


if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    # Generate random sample data
    src_data = torch.randint(
        1, src_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)
    tgt_data = torch.randint(
        1, tgt_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    transformer.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    transformer.eval()
    # Generate random sample validation data
    val_src_data = torch.randint(
        1, src_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(
            val_output.contiguous().view(-1, tgt_vocab_size),
            val_tgt_data[:, 1:].contiguous().view(-1),
        )
        print(f"Validation Loss: {val_loss.item()}")
