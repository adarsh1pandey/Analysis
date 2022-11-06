#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from difflib import SequenceMatcher


# In[2]:


lex = pd.read_csv('Hinglish_Profanity_List.csv', encoding='unicode_escape', header=None)


# In[3]:


hate_lexicons = []


# In[4]:


for word in lex[0]:
    hate_lexicons.append(word)


# In[5]:


# len(hate_lexicons)


# In[6]:


# lex.head(n=100)


# In[7]:


hateset = set(hate_lexicons)


# In[8]:


lists = [
    ['jews','muslims'],
    ['jews','islamic'],
    ['jews','mulle'],
    ['mulle','machod'],
    ['mulle','manhoos'],
    ['mulle','chutiya'],
    ['sheikh','muslims'],
    ['farsi','punjabi'],
    ['mulle','bengali'],
    ['bangla','farsi'],
    ['arabi','sindhi'],
    ['jaat','bangla'],
    ['sheikh','kashmiris'],
    ['sheikh','arabi'],
    ['sheikh','jaat'],
    ['maulvi','brahman'],
    ['aadmi', 'aurat'],
    ['kutte', 'kutiya'],
    ['ladka','ladki'],
    ['mulle','mulli'],
    ['mullah','brahman'],
    ['momedan','hindu'],
    ['kutta','kutti'],
    ['kutti', 'kutiya'],
    ['mohammad', 'ram'],
    ['kamina', 'kamini'],
    ['saala', 'saali'],
    ['haraam', 'haraami'],
    ['chupa', 'chupi'],
    ['madarchod', 'bhagatchod'],
    ['madarchod', 'fakeerchod'],
    ['bhagatchod', 'parichod'],
    ['madarchod', 'patichod'],
    ['behenchod', 'patichod'],
    ['behenchod', 'khandanchod'],
    ['musalmaan','mulli'],
    ['jihad', 'jihadi']
]


# In[9]:


ctr=0
fin_list = []
temp = []
for i in range(len(lists)):
    flag = False
    for j in range(len(lists[i])):
        if lists[i][j] in hateset:
            flag = True
            ctr+=1
    if flag:
        fin_list.append(lists[i])
        temp.append(lists[i][0])
        temp.append(lists[i][1])

print(ctr)


# In[10]:


# len(fin_list)


# In[11]:


temp = set(temp)


# In[12]:


hate_lexicons_init = [i for i in hate_lexicons if i not in temp]


# In[13]:


# len(hate_lexicons_init)


# In[14]:


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from nltk.tokenize import RegexpTokenizer
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.feature_extraction import text as sktext
from sklearn import preprocessing as skp
from keras import callbacks as kc
from keras import optimizers as ko
from keras import initializers, regularizers, constraints
from keras.engine import Layer
import keras.backend as K
from sklearn.metrics import f1_score
from keras.utils.vis_utils import model_to_dot
from keras import models
from keras import layers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten, GlobalAveragePooling1D, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import Constant


# In[15]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[16]:


df = pd.read_csv('final_data.csv')


# In[17]:


df.head()


# In[18]:


df['label'] = df['label'].astype(int)


# In[19]:


#df = df.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)


# In[20]:


train_df = pd.DataFrame()
train_df['content'] = df['lemmas']
train_df['class'] = df['label']


# In[21]:


# print("Training data phrase length distribution")
# sns.distplot(train_df['content'].map(lambda ele: len(ele)), kde_kws={"label": "train"})


# In[22]:


print('Most frequent sentence length in training:')
lens = train_df['content'].map(lambda ele: len(ele))
counts = np.bincount(lens)
print(np.argmax(counts))


# In[23]:


# print("Testing data phrase length distribution")
# sns.distplot(train_df['content'].map(lambda ele: len(ele)), kde_kws={"label": "test"})


# In[24]:


test_df = train_df[2800:]
train_df = train_df[:2800]


# In[25]:


# print("Testing data phrase length distribution")
# sns.distplot(test_df['content'].map(lambda ele: len(ele)), kde_kws={"label": "test"})


# In[26]:


print('Most frequent sentence length in testing:')
lens = test_df['content'].map(lambda ele: len(ele))
counts = np.bincount(lens)
print(np.argmax(counts))


# In[27]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# In[28]:



## A dictionary to map the punctuations present in the text to relevant strings or symbols
punct_mapping = {"‘": "'",
                 "₹": "e",
                 "´": "'",
                 "°": "",
                 "€": "e",
                 "™": "tm",
                 "√": " sqrt ",
                 "×": "x",
                 "²": "2",
                 "—": "-",
                 "–": "-",
                 "’": "'",
                 "_": "-",
                 "`": "'",
                 '“': '"',
                 '”': '"',
                 '“': '"',
                 "£": "e",
                 '∞': 'infinity',
                 'θ': 'theta',
                 '÷': '/',
                 'α': 'alpha',
                 '•': '.',
                 'à': 'a',
                 '−': '-',
                 'β': 'beta',
                 '∅': '',
                 '³': '3',
                 'π': 'pi',
                 ',':'',
                 '.':'',
                 ':':'',
                 '(':'',
                 ')':'',
                 '*':'',
                '"':'',
                '<':'',
                '>':''}


# In[29]:


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

## Function to remove special characters from the sentences (if any present)
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


# In[30]:


train_df.loc[:, 'content'] = train_df['content'].map(lambda text: clean_special_chars(text, punct, punct_mapping))
test_df.loc[:, 'content'] = test_df['content'].map(lambda text: clean_special_chars(text, punct, punct_mapping))


# In[31]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",
                   "can't've": "cannot have", "'cause": "because", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have","I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have","i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have","it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is",
                   "there'd": "there would", "there'd've": "there would have","there's": "there is",
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }


# In[32]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[33]:


train_df.loc[:, 'content'] = train_df['content'].map(lambda text: clean_contractions(text, contraction_mapping))
test_df.loc[:, 'content'] = test_df['content'].map(lambda text: clean_contractions(text, contraction_mapping))


# In[34]:


train_df.loc[:, 'content'] = train_df['content'].map(lambda text: text.lower())
test_df.loc[:, 'content'] = test_df['content'].map(lambda text: text.lower())


# In[35]:


data = train_df.values
data_test = test_df.values


# In[36]:


X_train = data[:,0]
Y_train = data[:,1]

X_test = data_test[:,0]
Y_test = data_test[:,1]

print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
print ("-------------------------")
print (X_test[0], Y_test[0])
print (X_train[0], Y_train[0])


# In[37]:


tokenizer = RegexpTokenizer("[a-zA-Z]+")
lemmatizer = WordNetLemmatizer()


# In[38]:


for ix in range(X_train.shape[0]):
    X_train[ix] = tokenizer.tokenize(X_train[ix])
    X_train[ix] = [lemmatizer.lemmatize(i) for i in X_train[ix]]

for ix in range(X_test.shape[0]):
    X_test[ix] = tokenizer.tokenize(X_test[ix])
    X_test[ix] = [lemmatizer.lemmatize(i) for i in X_test[ix]]


# In[39]:



unique_words = set()
len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)
#     print(sent)

    if(len_max<len(sent)):
        len_max = len(sent)

## Length of the list of unique_words gives the no of unique words
print("Vocabulary Size:")
print(len(list(unique_words)))
print("Maximum length of sentence:")
print(len_max)


# In[40]:


tokenizer_keras = Tokenizer(num_words=len(list(unique_words)))
tokenizer_keras.fit_on_texts(list(X_train))
X_train = tokenizer_keras.texts_to_sequences(X_train)
X_test = tokenizer_keras.texts_to_sequences(X_test)

## Padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
## Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print(X_train.shape,X_test.shape)


# In[41]:


embeddings_index = dict()


# In[42]:


from gensim.models import Word2Vec
from gensim.test.utils import common_texts


# In[43]:


model = Word2Vec.load("word2vec.model")


# In[44]:


word_vectors = model.wv


# In[45]:


print(data[:,0])


# In[46]:


embeddings_index = dict()


# In[47]:


for sent in data[:,0]:
    for token in sent:
        word = token
        word = lemmatizer.lemmatize(word)
        try:
            coefs = np.asarray(word_vectors[word], dtype='float32')
#             print(coefs)
#             print(coefs.shape)
            embeddings_index[word] = coefs
        except:
            print(word)
            coefs = np.zeros((200,))
            embeddings_index[word] = coefs


# In[48]:


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # Compute the dot product between u and v
    dot = np.dot(u,v)
    # Compute the L2 norm of u
    norm_u = np.linalg.norm(u)

    # Compute the L2 norm of v
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity defined by formula (1)
    cosine_similarity = dot / (norm_u*norm_v)

    return cosine_similarity


# In[49]:


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    # Select word vector representation of "word". Use word_to_vec_map.
    e = word_to_vec_map[word]

    # Compute e_biascomponent using the formula give above.
    e_biascomponent = np.dot(e,g)*g/(np.linalg.norm(g)**2)

    # Neutralize e by substracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e - e_biascomponent

    return e_debiased


# In[50]:


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Step 1: Select word vector representation of "word". Use word_to_vec_map.
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2
    mu = (e_w1+e_w2)/2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = np.dot(mu,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
    mu_orth = mu-mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B
    e_w1B = np.dot(e_w1,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
    e_w2B = np.dot(e_w2,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above
    corrected_e_w1B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w1B-mu_B)/(abs(e_w1-mu_orth-mu_B))
    corrected_e_w2B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w2B-mu_B)/(abs(e_w2-mu_orth-mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections
    e1 = corrected_e_w1B+mu_orth
    e2 = corrected_e_w2B+mu_orth

    return e1, e2


# In[51]:


full_list = hate_lexicons_init
for pairs in lists:
    full_list.extend(pairs)


# In[52]:


sents = df['lemmas'].values


# In[53]:


# len(sents)


# In[54]:


final_feat = []
ctr = 0
for sent in sents:
#     print(sent)
    words = sent.split()
    cur_vec = []
    for ix in range(len(full_list)):
        temp = False
        for word in words:
#             print(word)
            if word[0]=='p':
                continue
            if SequenceMatcher(a=full_list[ix], b=word).ratio()>0.85:
                temp = True
                ctr += 1
        if temp:
            cur_vec.append(1)
        else:
            cur_vec.append(0)
    final_feat.append(cur_vec)
print(ctr)


# In[55]:


np.save('PVExpt.npy', full_list)


# In[56]:


abuse_full = np.array(final_feat)


# In[57]:


abuse_full.shape


# In[58]:


len(full_list)


# In[59]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


vocab_size=len(list(unique_words))
embed_size = 200
max_features = vocab_size + 1


# In[62]:


checkpoint=ModelCheckpoint('model_self_embedding_final.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# In[63]:


X_full = np.concatenate((X_train, X_test))
Y_full = np.concatenate((Y_train, Y_test))


# In[64]:


Y_full = np_utils.to_categorical(Y_full)


# In[65]:


from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_full, Y_full, random_state=3, test_size=0.2)


# In[66]:


file_path = "PVExpt.hdf5"
early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=10)
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')


# In[67]:


lists.reverse()


# In[68]:


# lists


# In[69]:


final_feat = np.array(final_feat)


# In[70]:


# final_feat.shape


# In[ ]:


ctr = 2
for pair in lists:

    try:
        w1 = pair[0]
        w2 = pair[1]
        g = embeddings_index[w2] - embeddings_index[w1]
        e1, e2 = equalize((w1, w2), g, embeddings_index)
        embeddings_index[w1] = e1
        embeddings_index[w2] = e2
    except:
        pass

    embedding_matrix = np.zeros((vocab_size+1, 200))
    for word, i in tokenizer_keras.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    text_input_layer = Input(shape=(len_max,), dtype='int32')
    doc_embedding   = Embedding(len(list(unique_words))+1,
                                200,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=len_max,
                                trainable=False)(text_input_layer)

    abuse_full = final_feat[:,:-ctr]
    ctr += 2

    print('PV: ', abuse_full.shape[1], 'DB:', ctr/2)

    convs = []
    filter_sizes = [4, 5]
    # filter_sizes = [5]

    l_conv = Conv1D(filters=128, kernel_size=[3], padding='valid', activation='relu')(doc_embedding)

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='valid', activation='relu')(l_conv)
        convs.append(l_conv)
    cnn_feature_maps = convs

    reg_drop = Dropout(0.3)(l_conv)
    # flat = Flatten()(reg_drop)
    sentence_encoder1 = Bidirectional(LSTM(128,return_sequences=True))(reg_drop)
    sentence_encoder2 = Bidirectional(LSTM(128,return_sequences=True))(sentence_encoder1)
    att = Attention(286)(sentence_encoder2)
    fc_layer =Dense(128, activation="relu")(att)
    output_layer = Dense(2,activation="softmax")(fc_layer)

    model_1 = Model(inputs=[text_input_layer], outputs=[output_layer])

    model_1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#     model_1.summary()

    output1 = model_1.layers[-2].output
    # dense2 = Dense(300, activation='relu', name='layer_2')(output1)
    input4 = Input(shape=(abuse_full.shape[1],))
    dense3 = Dense(50, activation='relu', name='layer_3')(input4)
    merged = concatenate([output1, dense3])
    pre_final = Dense(50, activation='relu', name='pre_final')(merged)
    output = layers.Dense(2, activation="softmax", name="softmax_layer0")(pre_final)

    model_fin = Model(inputs = [model_1.input, input4], outputs = [output])


    model_fin.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#     model_fin.summary()

    abuse_train, abuse_test, y_train, y_test = model_selection.train_test_split(abuse_full, Y_full, random_state=3, test_size=0.2)

    hist = model_fin.fit([x_train,abuse_train],y_train,validation_data=([x_test,abuse_test],y_test),
                epochs = 10, batch_size=16,shuffle=True,callbacks=[checkpoint,early_stopping])

    pred_val = model_fin.predict([x_test, abuse_test])

    pred_val = pred_val.argmax(axis=1)

    true = y_test.argmax(axis=1)

    true = y_test.argmax(axis=1)

    final = pd.DataFrame()
    final['true'] = true.astype(int)
    final['pred'] = pred_val.astype(int)


    with open('results_HS.txt', 'a') as f:
    	f.write(classification_report(final['true'], final['pred']))

    print(classification_report(final['true'], final['pred']))


# In[53]:





# In[ ]:
