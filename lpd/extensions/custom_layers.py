import torch
import torch.nn as nn
import math

class MatMul2D(nn.Module):
    def __init__(self, transpose_b, name=None):
        super(MatMul2D, self).__init__()
        #PARAMS
        self.transpose_b = transpose_b
        self.name = name if name else 'mat_mul'

    def forward(self, a, b):
        if self.transpose_b:
            return torch.matmul(a, b.transpose(-2,-1))
        return torch.matmul(a, b)

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True, activation=None, name=None):
        super(Dense, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.activation = activation
        self.name = name if name else 'dense'
        #LAYERS
        self.fc = nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.use_bias:
            nn.init.zeros_(self.fc.bias)

    def forward(self, inputs):
        x = self.fc(inputs)
        if self.activation:
            x = self.activation(x)
        return x

class Attention(nn.Module):
    """
        The architecture is based on the paper “Attention Is All You Need”
        Usage (1)
        It can be used as Attention in transformer if q,k,v share the same dimensions.
        
        Usage (2)
        It can also be used as a method to aggregate a group of vectors into 1 vector if q dimensions are (batch, 1, key_dim)
        that way, instead of using Sum, or Average, you can have a learnable query vector (or a few of them) that will learn the aggregation function.
        See example in lpd.examples.multiple_inputs.model, where we define external_query_attention like so:
        external_query_attention = Attention(key_dim=config.EMBEDDINGS_SIZE, use_query_dense=True)  

        Args:
        key_dim - as defined in the paper, the number of expected features in the encoder inputs
        use_query_dense - whether to pass q input into another Dense layer, mostly used in Usage (2), to
                          run q into a transformation that will transform it into the vector space of k and v
        name - optional, any string to describe this layer
    """
    def __init__(self, key_dim, use_query_dense=False, name=None):
        super(Attention, self).__init__()
        #PARAMS
        self.key_dim            = key_dim
        self.sqrt_key_dim       = key_dim ** 0.5
        self.use_query_dense    = use_query_dense
        self.name               = name if name else 'attention'
        #LAYERS
        self.mat_mul2d          = MatMul2D(transpose_b=False, name = f'{self.name}__MatMul2D')
        self.mat_mul2d_t        = MatMul2D(transpose_b=True, name = f'{self.name}__MatMul2DT')
        self.softmax_last_dim   = nn.Softmax(dim=-1)
        if self.use_query_dense:
            # SOMETIMES WE WANT TO GO THROUGH ANOTHER TRANSFORMATION BEFORE RUNNING THE QUERY,
            # FOR EXAMPLE, WHEN THIS IS USED AS A STANDALONE LAYER
            self.query_dense    = Dense(in_dim=self.key_dim, out_dim=self.key_dim, use_bias=False, activation=None, name = f'{self.name}__Dense')

    def forward(self, q,k,v, mask = None):
        # q:    (batch, ?, key_dim)             where "?" can be 1 or seq_len
        # k:    (batch, seq_len, key_dim)
        # v:    (batch, seq_len, key_dim)
        # mask: (batch, 1, seq_len)

        # APPLY ATTENTION:
        #                       (     Q * Kt     )
        #               softmax (   ----------   ) * V
        #                       (    sqrt(dk)    )

        if self.use_query_dense:
            q = self.query_dense(q)                                            # (batch, seq_len, key_dim)

        q_k = self.mat_mul2d_t(q, k)                                           # (batch, ?, seq_len)
        scores = q_k / self.sqrt_key_dim                                       # (batch, ?, seq_len)

        if mask is not None:
            mask_ready = torch.log(mask)                                       # (batch, 1, seq_len)
            scores = scores + mask_ready                                       # (batch, ?, seq_len) (+= is doing broadcasting)

        attention_weights = self.softmax_last_dim(scores)                      # (batch, ?, seq_len)
        
        attention_output = self.mat_mul2d(attention_weights, v)                # (batch, ?, key_dim)

        return attention_output                                                # (batch, ?, key_dim)

class AttentionHead(nn.Module):
    def __init__(self, in_dim, key_dim, name=None):
        super(AttentionHead, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.sqrt_key_dim = key_dim ** 0.5
        self.name = name if name else 'Attention-Head'

        #LAYERS
        self.query_dense  = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__Q-Dense')
        self.key_dense    = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__K-Dense')
        self.value_dense  = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__V-Dense')
        self.att          = Attention(self.key_dim, name = f'{self.name}__Attention')

    def forward(self, inputs, mask = None):     # inputs:(batch, seq_len, emb_size), mask:(batch, seq_len)
        q = self.query_dense(inputs)            # (batch, seq_len, key_dim)
        k = self.key_dense(inputs)              # (batch, seq_len, key_dim)
        v = self.value_dense(inputs)            # (batch, seq_len, key_dim)
        return self.att(q,k,v,mask)             # (batch, seq_len, key_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim,
                       key_dim,
                       out_dim,
                       num_heads,
                       drop_out_proba,
                       name=None):
        super(MultiHeadAttention, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.drop_out_proba = drop_out_proba
        self.name = name if name else 'Multi-Head-Attention'

        #LAYERS
        self.attention_heads = nn.ModuleList([AttentionHead(self.in_dim, self.key_dim, name = f'{self.name}__H{i}') for i in range(self.num_heads)])
        self.output_dense = Dense(in_dim=self.num_heads*self.key_dim ,out_dim=self.out_dim, use_bias=True, activation=None, name = f'{self.name}__Out-Dense')
        self.dropout = nn.Dropout(p=self.drop_out_proba)
        self.norm = nn.LayerNorm(normalized_shape=self.out_dim) # WILL APPLY NORM OVER THE LAST DIMENTION ONLY

    def forward(self, inputs, mask=None):                                                     # inputs.shape: (batch, seq_len, emb_size == out_dim)
        attention_outputs = [head(inputs, mask = mask) for head in self.attention_heads]      # [ (batch, seq_len, key_dim) ]
        concatenated = torch.cat(attention_outputs, dim=-1)                                   # (batch, seq_len, key_dim * num_heads)
        output = self.output_dense(concatenated)                                              # (batch, seq_len, out_dim)
        output = self.dropout(output)                                                         # (batch, seq_len, out_dim)
        return self.norm(inputs + output)                             # RESIDUAL & NORM       # (batch, seq_len, out_dim)

class TransformerEncoderFeedForward(nn.Module):
    def __init__(self, in_dim,
                       out_dim,
                       drop_out_proba,
                       expansion_rate,
                       name=None):
        super(TransformerEncoderFeedForward, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop_out_proba = drop_out_proba
        self.expansion_rate = expansion_rate
        self.name = name if name else 'Transformer-Encoder__Feed-Forward'

        #LAYERS
        self.hidden_dense = Dense(in_dim=self.in_dim, out_dim=self.out_dim * self.expansion_rate, use_bias=True, activation=nn.ReLU(), name = f'{self.name}__Hidden-Dense')
        self.output_dense = Dense(in_dim=self.out_dim * self.expansion_rate, out_dim=self.out_dim, use_bias=True, activation=None, name = f'{self.name}__Out-Dense')
        self.dropout = nn.Dropout(p=self.drop_out_proba)

        self.norm = nn.LayerNorm(normalized_shape=self.out_dim)   # WILL APPLY NORM OVER THE LAST DIMENSION ONLY

    def forward(self, inputs):                                              # (batch, seq_len, out_dim)
        hidden_values = self.hidden_dense(inputs)                           # (batch, seq_len, out_dim * expansion_rate)
        output = self.output_dense(hidden_values)                           # (batch, seq_len, out_dim)
        output = self.dropout(output)                                       # (batch, num_elements, out_dim)
        return self.norm(inputs + output)    #RESIDUAL & NORM               # (batch, num_elements, out_dim)

class TransformerBlock(nn.Module):
    def __init__(self, in_dim,
                       key_dim,
                       out_dim,
                       num_heads,
                       drop_out_proba,
                       ff_expansion_rate,
                       name = None):
        super(TransformerBlock, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.drop_out_proba = drop_out_proba
        self.ff_expansion_rate = ff_expansion_rate
        self.name = name if name else 'Transformer-Encoder'
        #LAYERS
        self.multi_head_self_attention = MultiHeadAttention(self.in_dim,
                                                            self.key_dim,
                                                            self.out_dim,
                                                            self.num_heads,
                                                            self.drop_out_proba,
                                                            name = f'{self.name}__MH')

        self.feed_forward = TransformerEncoderFeedForward(self.out_dim,
                                                          self.out_dim,
                                                          self.drop_out_proba,
                                                          self.ff_expansion_rate,
                                                          name = f'{self.name}__FF')


    def forward(self, inputs, mask=None):                                           # inputs: (batch, num_elements, emb_size)
        attended = self.multi_head_self_attention(inputs=inputs, mask=mask)         # (batch, num_elements, out_dim)
        feed_forward = self.feed_forward(inputs=attended)                            # (batch, num_elements, out_dim)
        return feed_forward

class PositionalEncoding(nn.Module):
    #COPIED FROM https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, embedding_size, dropout_rate=0.1, maximum_position_encoding=5000):
        super(PositionalEncoding, self).__init__()
        #PARAMS
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.maximum_position_encoding = maximum_position_encoding
        #LAYERS
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self._create_positional_encoding()

    def _create_positional_encoding(self):
        if self.maximum_position_encoding:
            position = torch.arange(self.maximum_position_encoding).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.embedding_size, 2) * (-math.log(10000.0) / self.embedding_size))
            pe = torch.zeros(self.maximum_position_encoding, 1, self.embedding_size)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.squeeze(1)
            self.register_buffer('pe', pe)

    def forward(self, inputs):
        inputs = inputs + self.pe[:inputs.size(1), :].unsqueeze(0)
        return self.dropout(inputs)

class TransformerEncoderStack(nn.Module):
    def __init__(self, in_dim,
                       key_dim,
                       out_dim,
                       num_encoders,
                       num_heads,
                       drop_out_proba,
                       ff_expansion_rate,
                       maximum_position_encoding=None,
                       name=None):
        super(TransformerEncoderStack, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.drop_out_proba = drop_out_proba
        self.ff_expansion_rate = ff_expansion_rate
        self.maximum_position_encoding = maximum_position_encoding
        self.name = name if name else 'Transformer-Encoder-Stack'

        #LAYERS
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.in_dim,
                                                    self.key_dim,
                                                    self.out_dim,
                                                    self.num_heads,
                                                    self.drop_out_proba,
                                                    self.ff_expansion_rate,
                                                    name=f'{self.name}__E{i}')
                                                    for i in range(self.num_encoders)])
        self.pos_encoder = PositionalEncoding(self.in_dim, self.drop_out_proba, self.maximum_position_encoding)

    def forward(self, inputs, mask=None):
        outputs = inputs                                        # (batch, seq_len, emb_size)

        #POSITION
        if self.maximum_position_encoding is not None:
            outputs = self.pos_encoder(outputs)

        for encoder_layer in self.transformer_blocks:
            outputs = encoder_layer(inputs=outputs, mask=mask)
        return outputs                                # (batch, seq_len, out_dim)   <-- USUALLY out_dim = emb_size

