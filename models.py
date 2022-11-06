from email.mime import base
import numpy as np
import keras.backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine import Layer
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, LSTM
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten, GlobalAveragePooling1D, Reshape, concatenate
from keras.initializers import Constant
from keras.models import Model

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

class BaseModel(object):
    def __init__(self, unique_words, embed_size, len_max, embedding_matrix):
        self.unique_words = unique_words
        self.vocab_size=len(list(unique_words))
        self.embedding_matrix = embedding_matrix
        self.embed_size = embed_size
        self.max_features = self.vocab_size + 1
        self.len_max = len_max

    def build(self):
        text_input_layer = Input(shape=(self.len_max,), dtype='int32')
        doc_embedding   = Embedding(len(list(self.unique_words))+1,
                                    self.embed_size,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.len_max,
                                    trainable=False)(text_input_layer)

        convs = []
        filter_sizes = [4, 5]

        l_conv = Conv1D(filters=128, kernel_size=[3], padding='valid', activation='relu')(doc_embedding)

        for filter_size in filter_sizes:
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='valid', activation='relu')(l_conv)
            convs.append(l_conv)
        cnn_feature_maps = convs

        reg_drop = Dropout(0.3)(l_conv)
        sentence_encoder1 = Bidirectional(LSTM(128,return_sequences=True))(reg_drop)
        sentence_encoder2 = Bidirectional(LSTM(128,return_sequences=True))(sentence_encoder1)
        att = Attention()(sentence_encoder2)
        fc_layer =Dense(128, activation="relu")(att)
        output_layer = Dense(2,activation="softmax")(fc_layer)

        base_model = Model(inputs=[text_input_layer], outputs=[output_layer])
        return base_model

class AbuseModel(object):
    def __init__(self, base_model):
        self.base_model = base_model

    def build(self):
        output1 = self.base_model.layers[-2].output
        # dense2 = Dense(300, activation='relu', name='layer_2')(output1)
        input4 = Input(shape=(209,))
        dense3 = Dense(50, activation='relu', name='layer_3')(input4)
        merged = concatenate([output1, dense3])
        pre_final = Dense(50, activation='relu', name='pre_final')(merged)
        output = layers.Dense(2, activation="softmax", name="softmax_layer0")(pre_final)
        graph_model = Model(inputs = [self.base_model.input, input4], outputs = [output])
        return graph_model

class GraphModel(object):
    def __init__(self, base_model):
        self.base_model = base_model

    def build(self):
        output1 = self.base_model.layers[-2].output
        # dense2 = Dense(300, activation='relu', name='layer_2')(output1)
        input4 = Input(shape=(128,))
        dense3 = Dense(50, activation='relu', name='layer_3')(input4)
        merged = concatenate([output1, dense3])
        pre_final = Dense(50, activation='relu', name='pre_final')(merged)
        output = layers.Dense(2, activation="softmax", name="softmax_layer0")(pre_final)
        graph_model = Model(inputs = [self.base_model.input, input4], outputs = [output])
        return graph_model