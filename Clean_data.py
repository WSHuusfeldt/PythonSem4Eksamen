import pandas as pd
import pickle
import re
import string
import nltk
from nltk.corpus import wordnet
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import scattertext as st
import spacy
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF



democrats = ["Pete Buttigieg", "Joe Biden", "Amy Klobuchar", "Bernie Sanders", "Elizabeth Warren", "Andrew Yang"]
add_stop_words = ['just', 'like', 'got', 'things', 'thing', 'thats', 'know', 'said', 'going', 'dont', 'sure', 'mr', 'let', 'gon', 'na', 'say', 'want', 'year', 'time', 'end', 'way', 'talk', 'ive', 'im', 'tell', 'think', 'lot', 'mean', 'day', 'make', 'wait', 'right', 'youre', 'come', 'bring', 'theyre', 'ready', 'yeah', 'yes', 'buttigieg', 'klobuchar', 'yang', 'sander', 'warren', 'biden', 'people', 'country', 'oh', 'in', 'aa']

# Reads all the pickles of the democrats and puts the text into a list
def read_pickle_transcript():
    data = {}
    for i, c in enumerate(democrats):
        with open("transcripts/" + c + ".txt", "rb") as file:
            data[c] = pickle.load(file)
    return data
    
# Combies all the text
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ''.join(list_of_text)
    return combined_text

# Creates a dataframe of the transcripts
def create_dataframe_transcripts(data):
    data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
    pd.set_option('max_colwidth',150)
    democrats.sort()

    df = pd.DataFrame.from_dict(data_combined).transpose()
    df.columns = ['transcript']
    df = df.sort_index()
    df['politician'] = democrats
    return df

# "Cleans" the data (removes unnecessary stuff)
def clean_data(text):
    """ Clean data part 1: Lower case,  """
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', '', text)
    text = re.sub('\b', '', text)
    text = re.sub('[^a-z ]+', '', text)
    text = re.sub('\s\s+', ' ', text)
    return text

# For lemmatization we use WordNet, but we need to POS-tag the tokenized 
# words for a more accurate lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Lemmatizes a transcript
def lemmatize_transcript(text):
    ''' Tokenizes text and for each tokenized word, it applies a POS-tag and lemmatizes the word
        Returns the lemmatized output as a string
    '''
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(text)])
    return lemmatized_output

# Puts the data in a dataframe
def put_in_dataframe(data_combined):
    pd.set_option('max_colwidth',150)
    democrats.sort()
    df = pd.DataFrame.from_dict(data_combined).transpose()
    df.columns = ['transcript']
    df = df.sort_index()
    df['politician'] = democrats
    return df

# Converts from corpus to pandas dataframe
def convert_dataframe(cv, stop_words, data_clean_lemma):
    data_cv = cv.fit_transform(data_clean_lemma.transcript)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean_lemma.index
    return data_dtm

# Reads the pickle dtm.pkl
def read_data():
    data = pd.read_pickle('dtm.pkl')
    data = data.transpose()
    return data

def run():
    # Sets the data from the pickle
    data = read_pickle_transcript()

    # Combines the data
    data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

    # Data is put in a pandas dataframe, this form is called Corpus
    df = put_in_dataframe(data_combined)

    # Cleans data
    data_clean = pd.DataFrame(df.transcript.apply(lambda x: clean_data(x)))

    # Lemmatizes the transcript
    data_clean_lemma = pd.DataFrame(data_clean.transcript.apply(lambda x: lemmatize_transcript(x)))

    # Adds the stop words manually
    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

    # Vectorizes the stop words
    cv = CountVectorizer(stop_words=stop_words)
    
    # Sets the format from corpus to another dataframe (1 coloumn for each word)
    data_dtm = convert_dataframe(cv, stop_words, data_clean_lemma)

    # Pickle the data
    data_dtm.to_pickle("dtm.pkl")

    # Pickle data_clean_lemme
    data_clean_lemma.to_pickle('data_clean_lemma.pkl')
    pickle.dump(cv, open("cv.pkl", "wb"))

    # Reads the data from the pickle
    data = read_data()
    data_clean_lemma = pd.read_pickle('data_clean_lemma.pkl')


    
run()