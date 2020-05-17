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

def read_pickle_transcript():
    data = {}
    for i, c in enumerate(democrats):
        with open("transcripts/" + c + ".txt", "rb") as file:
            data[c] = pickle.load(file)
    return data

def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ''.join(list_of_text)
    return combined_text

def create_dataframe_transcripts(data):
    data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
    pd.set_option('max_colwidth',150)
    democrats.sort()

    df = pd.DataFrame.from_dict(data_combined).transpose()
    df.columns = ['transcript']
    df = df.sort_index()
    df['politician'] = democrats
    return df

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

def lemmatize_transcript(text):
    ''' Tokenizes text and for each tokenized word, it applies a POS-tag and lemmatizes the word
        Returns the lemmatized output as a string
    '''
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(text)])
    return lemmatized_output

def put_in_dataframe(data_combined):
    pd.set_option('max_colwidth',150)
    democrats.sort()
    df = pd.DataFrame.from_dict(data_combined).transpose()
    df.columns = ['transcript']
    df = df.sort_index()
    df['politician'] = democrats
    return df

def create_word_cloud(stop_words, data, data_clean_lemma):
    wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
                max_font_size=150, random_state=42)
    plt.rcParams['figure.figsize'] = [25 , 10]
    for index, democrat in enumerate(data.columns):
        wc.generate(data_clean_lemma.transcript[democrat])
        
        plt.subplot(3, 4, index+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(democrats[index])
    return plt

def create_top_dict(data):
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))
    return top_dict

def convert_dataframe(cv, stop_words, data_clean_lemma):
    data_cv = cv.fit_transform(data_clean_lemma.transcript)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean_lemma.index
    return data_dtm

def read_data():
    data = pd.read_pickle('dtm.pkl')
    data = data.transpose()
    return data

def create_scatter_plot(data_clean_lemma):
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [25 , 10]
    ax1 = fig.add_subplot(111)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Amy Klobuchar'],y=data_clean_lemma['polarity']['Amy Klobuchar'], label="Amy Klobuchar", s=400)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Andrew Yang'],y=data_clean_lemma['polarity']['Andrew Yang'], label="Andrew Yang", s=400)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Bernie Sanders'],y=data_clean_lemma['polarity']['Bernie Sanders'], label="Bernie Sanders", s=400)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Elizabeth Warren'],y=data_clean_lemma['polarity']['Elizabeth Warren'], label="Elizabeth Warren", s=400)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Joe Biden'],y=data_clean_lemma['polarity']['Joe Biden'], label="Joe Biden", s=400)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Pete Buttigieg'],y=data_clean_lemma['polarity']['Pete Buttigieg'], label="Pete Buttigieg", s=400)
    plt.legend(loc='upper right')
    plt.xlabel("Subjectivity")
    plt.ylabel("Polarity")
    return plt

def analyse_polarity_subjectivity(data_clean_lemma):
    data_clean_lemma['polarity'] = data_clean_lemma['transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data_clean_lemma['subjectivity'] = data_clean_lemma['transcript'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return data_clean_lemma




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

    # Adds the subjectivity and polarity
    data_clean_lemma = analyse_polarity_subjectivity(data_clean_lemma)

    # Puts the 30 most used words in a dictionary
    #top_dict = create_top_dict(data)

    # Sets up the word cloud
    plt = create_word_cloud(stop_words, data, data_clean_lemma)
    plt.show()

    # Sets up scatter plot
    plt = create_scatter_plot(data_clean_lemma)
    plt.show()

    
run()