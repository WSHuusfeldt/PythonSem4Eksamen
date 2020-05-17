import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import scattertext as st
import spacy
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.feature_extraction import text 
from collections import Counter


democrats = ["Amy Klobuchar", "Andrew Yang", "Bernie Sanders", "Elizabeth Warren", "Joe Biden", "Pete Buttigieg"]
add_stop_words = ['just', 'like', 'got', 'things', 'thing', 'thats', 'know', 'said', 'going', 'dont', 'sure', 'mr', 'let', 'gon', 'na', 'say', 'want', 'year', 'time', 'end', 'way', 'talk', 'ive', 'im', 'tell', 'think', 'lot', 'mean', 'day', 'make', 'wait', 'right', 'youre', 'come', 'bring', 'theyre', 'ready', 'yeah', 'yes', 'buttigieg', 'klobuchar', 'yang', 'sander', 'warren', 'biden', 'people', 'country', 'oh', 'in', 'aa']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

def run(): 
    # Reads the data from the pickle
    data = pd.read_pickle('dtm.pkl')
    data = data.transpose()
    data_clean_lemma = pd.read_pickle('data_clean_lemma.pkl')
    # Puts the 30 most used words in a dictionary
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))

    # Sets up the word cloud
    wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
                max_font_size=150, random_state=42)
    plt.rcParams['figure.figsize'] = [25 , 10]
    for index, democrat in enumerate(data.columns):
        wc.generate(data_clean_lemma.transcript[democrat])
        
        plt.subplot(3, 4, index+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(democrats[index])
    plt.show()

    print(data_clean_lemma)
    #Loads the english Language
    nlp = spacy.load('en_core_web_sm')
    #Creates colon politician
    data_clean_lemma['politician'] = democrats
    #Sets the transcript up.
    corpus = st.CorpusFromPandas(data_clean_lemma, category_col='politician', text_col='transcript', nlp=nlp).build()
    

    # Term frequency - bigram
    def term_freq(politician):
        term_freq_df = corpus.get_term_freq_df()
        term_freq_df['Score'] = corpus.get_scaled_f_scores(politician)
        print(term_freq_df.sort_values(by='Score', ascending=False).index[:20])

    for x in democrats: 
        print(x)  
        term_freq(x)
        print('---------')
    
    #Unique words and Vocabulary
    list_words = []
    total_list = []

    for dem in data.columns:
        uniques = data[dem].to_numpy().nonzero()[0].size
        list_words.append(uniques)
        totals = sum(data[dem])
        total_list.append(totals)

    data_words = pd.DataFrame(list(zip(democrats, list_words)), columns=['politician', 'unique_words'])
    data_unique_sort = data_words.sort_values(by='unique_words', ascending=False)
    data_words['total_words'] = total_list
    data_total_sort = data_words.sort_values(by='total_words', ascending=False)
    print(data_total_sort)

    #Graph of unique and total words
    y_pos = np.arange(len(data_words))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.barh(y_pos, data_unique_sort.unique_words, align='center')
    plt.yticks(y_pos, data_unique_sort.politician)
    plt.title('Number of Unique Words', fontsize=20)

    plt.subplot(1, 2, 2)
    plt.barh(y_pos, data_total_sort.total_words, align='center')
    plt.yticks(y_pos, data_total_sort.politician)
    plt.title('Number of Total words', fontsize=20)

    plt.tight_layout()
    plt.show()

    # Trying to Topic Modeling 
    data_dtm = pd.read_pickle('dtm.pkl')
    cv = pd.read_pickle('cv.pkl')
    data_cv = cv.fit_transform(data_clean_lemma.transcript)
    lda_model = LatentDirichletAllocation(n_components=5, learning_method='online', max_iter=50, random_state=0).fit(data_dtm)

    no_top_words = 10

    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print('Topic %d:' % (topic_idx))
            print(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words -1:-1]]))

    fn = cv.get_feature_names()
    display_topics(lda_model, fn, no_top_words)

    # # using sklearn and pyLDAvis we see that it only found 2 topics, and those being ambiguous or rather large. The problem could be the amount of data used, but also the sircumstance
    # # of the speeches. A really hot topic of the speeches are about healthcare, which is most likely the number 1 topic.

    
    # panel = pyLDAvis.sklearn.prepare(lda_model, data_cv, cv, mds='tsne')
    # pyLDAvis.show(panel)

    # Creation of vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, use_idf=True)

    # Fit on our data
    tfidf = tfidf_vectorizer.fit_transform(data_clean_lemma['transcript'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    #makes a dataframe over all the words used by each politician
    dtm_tfidf = pd.DataFrame(tfidf.toarray(), columns=list(tfidf_feature_names))
    dtm_tfidf.index = data_clean_lemma.index
    print(dtm_tfidf)

    #prepares data to generate topics
    nmf = NMF(n_components=6, random_state=0, alpha=.1).fit(dtm_tfidf)
    display_topics(nmf, tfidf_feature_names, no_top_words)

    #Sets topic values from the nmf model
    nmf_topic_values = nmf.transform(tfidf)
    #sets topic values in nmf_topic column
    data_clean_lemma['nmf_topics'] = nmf_topic_values.argmax(axis=1)

    #Creates a dictionary over the topics genres and puts them together with the topics
    nmf_remap = {0: 'Healthcare/Medicare', 1: 'Education', 2: 'Middle class/Foreign Policy', 3: 'Democracy/Freedom', 4: 'Family/Community', 5: 'Working class'}
    data_clean_lemma['nmf_topics'] = data_clean_lemma['nmf_topics'].map(nmf_remap)
    # print(data_clean_lemma)

    print("-----------------------")

    data_clean_lemma['polarity'] = data_clean_lemma['transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data_clean_lemma['subjectivity'] = data_clean_lemma['transcript'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    print(data_clean_lemma)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x=data_clean_lemma['subjectivity']['Amy Klobuchar'],y=data_clean_lemma['polarity']['Amy Klobuchar'], label="Amy Klobuchar", s=150)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Andrew Yang'],y=data_clean_lemma['polarity']['Andrew Yang'], label="Andrew Yang", s=150)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Bernie Sanders'],y=data_clean_lemma['polarity']['Bernie Sanders'], label="Bernie Sanders", s=150)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Elizabeth Warren'],y=data_clean_lemma['polarity']['Elizabeth Warren'], label="Elizabeth Warren", s=150)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Joe Biden'],y=data_clean_lemma['polarity']['Joe Biden'], label="Joe Biden", s=150)
    ax1.scatter(x=data_clean_lemma['subjectivity']['Pete Buttigieg'],y=data_clean_lemma['polarity']['Pete Buttigieg'], label="Pete Buttigieg", s=150)
    plt.legend(loc='upper right')
    plt.xlabel("Subjectivity")
    plt.ylabel("Polarity")
    plt.show()
run()