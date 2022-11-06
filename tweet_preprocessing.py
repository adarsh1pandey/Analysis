import numpy as np
import pandas as pd
import json
import preprocessor as p
import nltk
import re
from nltk import word_tokenize, FreqDist
from nltk.corpus import words
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
from nltk.tokenize import TweetTokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from google_trans_new.google_trans_new import google_translator  

class TweetPreprocessor(object):
    def __init__(self):
        self.df = pd.read_csv('./Data/tweet_data.csv')
        with open('./Data/tweet_data.json', 'r') as f:
            self.tweet_map = json.load(f)
        self.punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        ## A dictionary to map the punctuations present in the text to relevant strings or symbols
        self.punct_mapping = {
            "‘": "'", 
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
            '>':''
        }
        self.contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
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
        self.tweet_tokenizer = TweetTokenizer()
        self.translator= google_translator()


    def remove_digits_lower_case(self, tweets):
        lower_case_tweets = []
        for tweet in tweets:
            tweet = re.sub(r'[0-9]+', '', tweet)
            lower_case_tweets.append(tweet.lower())
        return lower_case_tweets

    def clean_code_mixed_tweets(self, tweets):
        clean_code_mixed_tweets = []
        for tweet in tweets:
            try:
                clean_code_mixed_tweets.append(p.clean(tweet))
            except:
                clean_code_mixed_tweets.append('')
        return clean_code_mixed_tweets

    ## Function to remove special characters from the sentences (if any present)
    def clean_special_chars(self, tweets):
        clean_special_chars_tweets = []
        for tweet in tweets:
            for p in self.punct_mapping:
                tweet = tweet.replace(p, self.punct_mapping[p])
            
            specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
            for s in specials:
                tweet = tweet.replace(s, specials[s])
            clean_special_chars_tweets.append(tweet)
        return clean_special_chars_tweets

    def clean_contractions(self, tweets):
        contraction_mapping_tweets = []
        for tweet in tweets:
            specials = ["’", "‘", "´", "`"]
            for s in specials:
                tweet = tweet.replace(s, "'")
            tweet = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in tweet.split(" ")])
            contraction_mapping_tweets.append(tweet)
        return contraction_mapping_tweets

    def transliteration_based_preprocessing(self, tweets):
        transliteration_based_preprocessed_tweets = []
        for tweet in tweets:
            tweet_tokens = self.tweet_tokenizer.tokenize(tweet)
            ctr = 0
            hindi_tweet = ''
            for tweet_token in tweet_tokens:
                if tweet_token in words.words():
                    hindi_tweet += self.translator.translate(tweet_token, lang_src='en', lang_tgt='hi')
                else:
                    hindi_tweet += transliterate(tweet_token, sanscript.ITRANS, sanscript.DEVANAGARI)
            english_tweet = self.translator.translate(hindi_tweet, lang_src='hi', lang_tgt='en')
            transliteration_based_preprocessed_tweets.append(english_tweet)
        return transliteration_based_preprocessed_tweets

    def fine_tune_word2vec(self, tweets):
        tweets_tokenized = [self.tweet_tokenizer.tokenize(i) for i in tweets]
        base_w2v_model = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin",
                                         binary = True)
        new_w2v_model = Word2Vec(size=300, min_count=1)
        new_w2v_model.build_vocab(tweets_tokenized)
        total_examples = new_w2v_model.corpus_count
        new_w2v_model.build_vocab([list(base_w2v_model.vocab.keys())], update=True)
        new_w2v_model.intersect_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)
        new_w2v_model.train(tweets_tokenized, total_examples=total_examples, epochs=new_w2v_model.iter)
        new_w2v_model.save("./word2vec/new_w2v_model.model")

    def map_users(self, tweet_ids):
        user_ids = []
        for tweet_id in tweet_ids:
            try:
                user_ids.append(self.tweet_map[tweet_id]['user']['id'])
            except:
                user_ids.append(0)
        self.df['user'] = user_ids

if __name__ == '__main__':
    tweet_preprocessor = TweetPreprocessor()
    tweets = tweet_preprocessor.df['tweet']
    tweet_ids = tweet_preprocessor.df['id']
    tweet_preprocessor.map_users(tweet_ids)
    clean_code_mixed_tweets = tweet_preprocessor.clean_code_mixed_tweets(tweets)
    lower_case_tweets = tweet_preprocessor.remove_digits_lower_case(clean_code_mixed_tweets)
    clean_special_chars_tweets = tweet_preprocessor.clean_special_chars(lower_case_tweets)
    contraction_mapping_tweets = tweet_preprocessor.clean_contractions(clean_special_chars_tweets)
    tweet_preprocessor.df['cleaned_tweets'] = contraction_mapping_tweets
    tweet_preprocessor.df.to_csv('./Data/final_data_1.csv')
    # transliteration_based_preprocessed_tweets = tweet_preprocessor.transliteration_based_preprocessing(contraction_mapping_tweets)
    # tweet_preprocessor.df['transliterated_tweets'] = transliteration_based_preprocessed_tweets
    # tweet_preprocessor.df.to_csv('final_data.csv')
    # print(tweet_preprocessor.df.head())
    # tweet_preprocessor.fine_tune_word2vec(contraction_mapping_tweets)