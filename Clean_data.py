import pandas as pd
import pickle
import re
import string

democrats = ["Pete Buttigieg", "Joe Biden", "Amy Klobuchar", "Bernie Sanders", "Elizabeth Warren", "Andrew Yang"]

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

data = read_pickle_transcript()
print(create_dataframe_transcripts(data))
    
