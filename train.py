import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from models import BaseModel, AbuseModel, GraphModel
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils


class Trainer(object):
    def __init__(self):
        self.unique_words = set()
        self.len_max = 0
        self.tokenizer = RegexpTokenizer("[a-zA-Z]+")
        self.word2vec_model = Word2Vec.load("./word2vec/new_w2v_model.model")
        self.word_vectors = self.word2vec_model.wv
        self.embeddings_index = dict()
        self.embed_size = 300
        self.file_path = "./Checkpoints/weights_base_[C-LSTM].best.hdf5"
        self.early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=10)
        self.checkpoint = ModelCheckpoint(self.file_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
        self.abuse_train = np.load('./Data/abuse_train.npy')
        self.abuse_test = np.load('./Data/abuse_test.npy')

    def load_preprocessed_data(self):
        self.df = pd.read_csv('./Data/final_data_1.csv')
        self.df.label = pd.Categorical(self.df.label)
        self.df['class'] = self.df.label.cat.codes
        self.df['content'] = self.df['cleaned_tweets']

    def split_train_test(self, test_size, x, y):
        return train_test_split(x, y, test_size=test_size, random_state=42)

    def build_vocab(self, x):
        for sent in tqdm(x):
            self.unique_words.update(sent)
            if(self.len_max<len(sent)):
                self.len_max = len(sent)
        print(len(self.unique_words), self.len_max)

    def tokenise_tweets(self, x):
        for ix in range(x.shape[0]):
            x[ix] = self.tokenizer.tokenize(str(x[ix]))
        return x

    def keras_preprocessing(self, x, x_test, x_train):
        self.tokenizer_keras = Tokenizer(num_words=len(list(self.unique_words)))
        self.tokenizer_keras.fit_on_texts(list(x))
        x_train = self.tokenizer_keras.texts_to_sequences(x_train)
        x_test = self.tokenizer_keras.texts_to_sequences(x_test)

        ## Padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
        ## Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.
        x_train = sequence.pad_sequences(x_train, maxlen=self.len_max)
        x_test = sequence.pad_sequences(x_test, maxlen=self.len_max)
        print(x_train.shape,x_test.shape)
        return x_train, x_test

    def prepare_embedding_index(self, x):
        for sent in x:
            for token in sent:
                word = token
                try:
                    coefs = np.asarray(self.word_vectors[word], dtype='float32')
                    self.embeddings_index[word] = coefs
                except:
                    # print(word)
                    coefs = np.zeros((self.embed_size,))
                    self.embeddings_index[word] = coefs

    def prepare_embedding_matrix(self):
        vocab_size=len(list(self.unique_words))
        self.embedding_matrix = np.zeros((vocab_size+1, self.embed_size))
        for word, i in self.tokenizer_keras.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        self.max_features = vocab_size + 1

    def read_user_embeddings(self, file_enc, skip_lines=0, filter_set=None):
        self.user_emb_dict = dict()
        total_vectors_in_file = 0
        with open('./Data/user_embeddings.emd', 'r') as f:
            for i, line in enumerate(f):
                if i < skip_lines:
                    continue
                if not line:
                    break
                if len(line) == 0:
                    # is this reachable?
                    continue

                l_split = line.strip().split(' ')
                if len(l_split) == 2:
                    continue
                total_vectors_in_file += 1
                if filter_set is not None and l_split[0] not in filter_set:
                    continue
                self.user_emb_dict[l_split[0]] = [float(em) for em in l_split[1:]]

    def prepare_user_data(self):
        user_embeddings = []
        for user in self.df['user'].values:
            try:
                user_embeddings.append(self.user_emb_dict[str(user)])
            except:
                user_embeddings.append(np.zeros(128,))
        self.df['user_emb'] = user_embeddings

    def get_base_model(self):
        base_model_builder = BaseModel(self.unique_words, self.embed_size, self.len_max, self.embedding_matrix)
        base_model = base_model_builder.build()
        base_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        base_model.summary()
        return base_model

    def get_abuse_model(self, base_model):
        abuse_model_builder = AbuseModel(base_model)
        abuse_model = abuse_model_builder.build()
        abuse_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        abuse_model.summary()
        return abuse_model

    def get_graph_model(self, base_model):
        graph_model_builder = GraphModel(base_model)
        graph_model = graph_model_builder.build()
        graph_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        graph_model.summary()
        return graph_model

    def train_model(self, model, x_train, x_test, y_train, y_test):
        hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),
                epochs = 10, batch_size=16,shuffle=True,callbacks=[self.early_stopping, self.checkpoint])


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_preprocessed_data()
    trainer.read_user_embeddings('./Data/user_embeddings.emd')
    trainer.prepare_user_data()
    x_train, x_test, y_train, y_test = trainer.split_train_test(0.33, trainer.df['content'], trainer.df['class'])
    user_train, user_test, y_train, y_test = trainer.split_train_test(0.33, np.array(trainer.df['user_emb'].tolist()), trainer.df['class'])
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, user_train.shape, user_test.shape)
    x_train = trainer.tokenise_tweets(x_train.values)
    x_test = trainer.tokenise_tweets(x_test.values)
    x_combined = np.append(x_train, x_test)
    trainer.build_vocab(x_combined)
    x_train, x_test = trainer.keras_preprocessing(x_combined, x_test, x_train)
    trainer.prepare_embedding_index(x_combined)
    trainer.prepare_embedding_matrix()
    base_model = trainer.get_base_model()
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    # trainer.train_model(base_model, x_train, x_test, y_train, y_test)
    abuse_model = trainer.get_abuse_model(base_model)
    # trainer.train_model(abuse_model, [x_train, trainer.abuse_train], [x_test, trainer.abuse_test], y_train, y_test)
    graph_model = trainer.get_graph_model(base_model)
    trainer.train_model(graph_model, [x_train, user_train], [x_test, user_test], y_train, y_test)



