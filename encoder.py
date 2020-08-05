import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization

class Encoder(layers.Layer):

    def __init__(self, vocab_size, max_len, embedding_dim=128, num_heads=4, ffn_units=512, dp_rate=0.1, encoders=4):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEmbedding(vocab_size, max_len, embedding_dim, dp_rate)
        self.encoder_stack = [EncoderLayer(embedding_dim, num_heads, ffn_units, dp_rate) for i in range(encoders)]

    def call(self, x, training):
        enc_mask = self.pos_emb.compute_mask(x)
        enc_output = self.pos_emb(x)
        for encoder in self.encoder_stack:
            enc_output = encoder(enc_output, training, enc_mask)
        return enc_output

class EncoderLayer(layers.Layer):

    def __init__(self, embedding_dim=128, heads=4, ffn_units=512, dp_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, heads)
        self.add_norm = Add_Normalization(dp_rate)
        self.pwffn = PointWiseFFN(ffn_units, embedding_dim, dp_rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, mask)
        add_norm_output = self.add_norm(attn_output, x, training)
        pwffn_output = self.pwffn(add_norm_output, training)
        return self.add_norm(pwffn_output, add_norm_output, training)


class PointWiseFFN(layers.Layer):

    def __init__(self, units, embedding_dim, dp_rate=0.1):
        super(PointWiseFFN, self).__init__()
        self.dropout = Dropout(dp_rate)
        self.relu_dense = Dense(units, 'relu')
        self.lin_dense = Dense(embedding_dim)

    def call(self, attn_output, training):
        attn_output = self.dropout(attn_output, training=training)
        attn_output = self.relu_dense(attn_output)
        return self.lin_dense(attn_output)


class Add_Normalization(layers.Layer):

    def __init__(self, dp_rate=0.1):
        super(Add_Normalization, self).__init__()
        self.layernorm = layers.LayerNormalization()
        self.dropout = Dropout(dp_rate)

    def call(self, attn_output, x, training):
        attn_output = self.dropout(attn_output, training=training)
        attn_output += x
        return self.layernorm(attn_output)


class MultiHeadAttention(layers.Layer):

    def __init__(self, embedding_dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.units = embedding_dim
        self.heads = heads
        self.hydra = self.multi_attention()
        self.W = Dense(embedding_dim, use_bias=False)

    def multi_attention(self):
        return [SelfAttention(self.depth()) for i in range(self.heads)]

    def depth(self):
        assert self.units % self.heads == 0
        return self.units // self.heads

    def call(self, embedded_seq, mask=None):
        attn_output = [head(embedded_seq, mask) for head in self.hydra]
        return self.W(layers.concatenate(attn_output, axis=2))


class SelfAttention(layers.Layer):

    def __init__(self, depth=32):
        super(SelfAttention, self).__init__()
        self.Q1 = Dense(depth, use_bias=False)
        self.K1 = Dense(depth, use_bias=False)
        self.V1 = Dense(depth, use_bias=False)
        self.depth = depth

    def call(self, embedded_seq, mask=None):
        Q1 = self.Q1(embedded_seq)
        K1 = self.K1(embedded_seq)
        V1 = self.V1(embedded_seq)
        prod = tf.matmul(Q1, tf.transpose(K1, [0, 2, 1]))
        scaled_attention_logits = prod / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        softmax = tf.nn.softmax(scaled_attention_logits, axis=-1)
        final_attention = tf.matmul(softmax, V1)
        return final_attention


class PositionalEmbedding(layers.Layer):
    '''
    the model itself learns the positional embedding
    '''

    def __init__(self, vocab_size, max_len, embedding_dim=128, dp_rate=0.1):
        super(PositionalEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=max_len, output_dim=embedding_dim, mask_zero=True)
        self.dropout = Dropout(dp_rate)

    def call(self, x, training):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_emb(positions)

        x = self.token_emb(x) + positions
        return self.dropout(x, training)

    def compute_mask(self, x, mask=None):
        a = tf.cast(self.token_emb.compute_mask(x), tf.float32)
        return 1 - a[:, tf.newaxis, :]