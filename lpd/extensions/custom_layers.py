import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math

class MatMul2D(nn.Module):   
    def __init__(self, transpose_b, name=None): 
        super(MatMul2D, self).__init__()
        #PARAMS
        self.transpose_b = transpose_b
        self.name = name if name else 'mat_mul'

    def forward(self, a, b):
        if self.transpose_b:
            return T.bmm(a, b.transpose(1,2))
        return T.bmm(a, b)

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
    def __init__(self, key_dim, use_query_dense=False, name=None):
        super(Attention, self).__init__()
        #PARAMS
        self.key_dim            = key_dim
        self.sqrt_key_dim       = key_dim ** 0.5
        self.use_query_dense    = use_query_dense
        self.name               = name if name else 'attention'
        #LAYERS
        self.mat_mul2d            = MatMul2D(transpose_b=False, name = f'{self.name}__MatMul2D')
        self.mat_mul2d_t          = MatMul2D(transpose_b=True, name = f'{self.name}__MatMul2DT')
        if self.use_query_dense:
            # SOMETIMES WE WANT TO GO TROUGH ANPTHER TRANSFORMATION BEFORE RUNNING THE QUERY, FOR EXAMPLE, WHEN THIS IS USED AS A STANDALONE LAYER
            self.query_dense    = Dense(in_dim=self.key_dim, out_dim=self.key_dim, use_bias=False, activation=None, name = f'{self.name}__Dense')

    def forward(self, q,k,v, mask = None):    
        # q:    (batch, ?, key_dim)             "?" can be 1 or num_elements
        # k:    (batch, num_elements, key_dim)
        # v:    (batch, num_elements, key_dim) 
        # mask: (batch, 1, num_elements)

        # APPLY ATTENTION:
        #                       (     Q * Kt     )  
        #               softmax (   ----------   ) * V
        #                       (    sqrt(dk)    )

        if self.use_query_dense:
            q = self.query_dense(q)                                            # (batch, num_elements, key_dim)

        q_k = self.mat_mul2d_t(q, k)                                             # (batch, ?, num_elements)
        scores = q_k / self.sqrt_key_dim                                       # (batch, ?, num_elements)

        if mask is not None:
            mask_ready = T.log(mask)                                           # (batch, 1, num_elements)
            scores += mask_ready                                               # (batch, ?, num_elements) (+= is doing broadcasting)

        attention_weights = F.softmax(scores, dim=-1)                          # (batch, ?, num_elements)
        attention_output = self.mat_mul2d(attention_weights, v)                  # (batch, ?, key_dim)

        return attention_output                                                # (batch, ?, key_dim)   

class AttentionHead(nn.Module):
    def __init__(self, in_dim, key_dim, name=None):
        super(AttentionHead, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.sqrt_key_dim = key_dim ** 0.5
        self.name = name if name else 'attention_head'

        #LAYERS
        self.query_dense  = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__Q-Dense')
        self.key_dense    = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__K-Dense')
        self.value_dense  = Dense(self.in_dim, self.key_dim, use_bias=True, activation=None, name = f'{self.name}__V-Dense')
        self.att          = Attention(self.key_dim, name = f'{self.name}__Attention') 

    def forward(self, inputs, mask = None):     # inputs:(batch, num_elements, emb_size), mask:(batch, num_elements)
        q = self.query_dense(inputs)            # (batch, num_elements, key_dim)  
        k = self.key_dense(inputs)              # (batch, num_elements, key_dim)  
        v = self.value_dense(inputs)            # (batch, num_elements, key_dim)  
        return self.att(q,k,v,mask)             # (batch, num_elements, key_dim)     

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
        self.name = name if name else 'multihead_attentions'

        #LAYERS
        self.attention_heads = nn.ModuleList([AttentionHead(self.in_dim, self.key_dim, name = f'{self.name}__H{i}') for i in range(self.num_heads)])
        self.output_dense = Dense(in_dim=self.num_heads*self.key_dim ,out_dim=self.out_dim, use_bias=True, activation=None, name = f'{self.name}__Out-Dense')
        self.dropout_inplace = nn.Dropout(p=self.drop_out_proba, inplace=True)
        self.norm = nn.LayerNorm(normalized_shape=self.out_dim) # WILL APPLY NORM OVER THE LAST DIMENTION ONLY

    def forward(self, inputs, mask=None):                                                     # inputs.shape: (batch, num_elements, emb_size == out_dim) 
        attention_outputs = [head(inputs, mask = mask) for head in self.attention_heads]   # [ (batch, num_elements, key_dim) ]
        concatanated = T.cat(attention_outputs, dim=-1)                                       # (batch, num_elements, key_dim * num_heads) 
        output = self.output_dense(concatanated)                                              # (batch, num_elements, out_dim) 
        self.dropout_inplace(output)                                                          # (batch, num_elements, out_dim)
        return self.norm(inputs + output)                             # RESIDUAL & NORM       # (batch, num_elements, out_dim)

class TransformerEncoderFeedForward(nn.Module):
    def __init__(self, in_dim,
                       out_dim,
                       drop_out_proba,
                       expantion_rate,
                       name=None):
        super(TransformerEncoderFeedForward, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop_out_proba = drop_out_proba
        self.expantion_rate = expantion_rate
        self.name = name if name else 'transformer_encoder__feed_forward'

        #LAYERS
        self.hidden_dense = Dense(in_dim=self.in_dim, out_dim=self.out_dim * self.expantion_rate,  use_bias=True, activation=F.relu, name = f'{self.name}__Hidden-Dense')
        self.output_dense = Dense(in_dim=self.out_dim * self.expantion_rate, out_dim=self.out_dim, use_bias=True, activation=None, name = f'{self.name}__Out-Dense')
        self.dropout_inplace = nn.Dropout(p=self.drop_out_proba, inplace=True)

        self.norm = nn.LayerNorm(normalized_shape=self.out_dim)   # WILL APPLY NORM OVER THE LAST DIMENTION ONLY

    def forward(self, inputs):                                              # (batch, num_elements, out_dim) 
        hidden_values = self.hidden_dense(inputs)                           # (batch, num_elements, out_dim * expantion_rate)
        output = self.output_dense(hidden_values)                           # (batch, num_elements, out_dim)
        self.dropout_inplace(output)                                        # (batch, num_elements, out_dim)
        return self.norm(inputs + output)    #RESIDUAL & NORM               # (batch, num_elements, out_dim)

class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, 
                       key_dim, 
                       out_dim,
                       num_heads, 
                       drop_out_proba,
                       ff_expantion_rate,
                       name = None):
        super(TransformerEncoder, self).__init__()
        #PARAMS
        self.in_dim = in_dim 
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.drop_out_proba = drop_out_proba
        self.ff_expantion_rate = ff_expantion_rate
        self.name = name if name else 'transformer_encoder'
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
                                                          self.ff_expantion_rate,
                                                          name = f'{self.name}__FF')


    def forward(self, inputs, mask=None):                                           # inputs: (batch, num_elements, emb_size)
        attended = self.multi_head_self_attention(inputs=inputs, mask = mask)       # (batch, num_elements, out_dim)
        fed_forward = self.feed_forward(inputs=attended)                            # (batch, num_elements, out_dim)
        return fed_forward

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
            pe = T.zeros(self.maximum_position_encoding, self.embedding_size)
            position = T.arange(0, self.maximum_position_encoding, dtype=T.float).unsqueeze(1)
            div_term = T.exp(T.arange(0, self.embedding_size, 2).float() * (-math.log(10000.0) / self.embedding_size))
            pe[:, 0::2] = T.sin(position * div_term)
            pe[:, 1::2] = T.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

    def forward(self, inputs):
        inputs = inputs + self.pe[:inputs.size(0), :]
        return self.dropout(inputs)

class TransformerEncoderStack(nn.Module):
    def __init__(self, in_dim,
                       key_dim, 
                       out_dim,
                       num_transformer_encoders,
                       num_heads_per_transformer, 
                       drop_out_proba,
                       ff_expantion_rate,
                       maximum_position_encoding=None,
                       name = None):
        super(TransformerEncoderStack, self).__init__()
        #PARAMS
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.num_transformer_encoders = num_transformer_encoders
        self.num_heads_per_transformer = num_heads_per_transformer
        self.drop_out_proba = drop_out_proba
        self.ff_expantion_rate = ff_expantion_rate
        self.maximum_position_encoding = maximum_position_encoding
        self.name = name if name else 'transformer_encoder_stack'

        #LAYERS
        self.encoder_layers = nn.ModuleList([TransformerEncoder(self.in_dim, 
                                                    self.key_dim,                          
                                                    self.out_dim,                         
                                                    self.num_heads_per_transformer,       
                                                    self.drop_out_proba,                  
                                                    self.ff_expantion_rate,               
                                                    name = f'{self.name}__E{i}')          
                                                    for i in range(self.num_transformer_encoders)])
        self.pos_encoder = PositionalEncoding(self.in_dim, self.drop_out_proba, self.maximum_position_encoding)

    def forward(self, inputs, mask=None):
        outputs = inputs                                        # (batch, num_elements, emb_size)

        #POSITION
        if self.maximum_position_encoding is not None:
            outputs = self.pos_encoder(outputs)

        for encoder_layer in self.encoder_layers:
            outputs = encoder_layer(inputs=outputs, mask=mask)    
        return outputs                                # (batch, num_elements, out_dim)   <-- USUALLY out_dim = emb_size






