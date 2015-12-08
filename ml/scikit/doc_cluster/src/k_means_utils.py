# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import nltk
import re

def get_k_means_cluster(tfidf_matrix, num_clusters):
    from sklearn.cluster import KMeans

    print ("Running k-means on " + str(num_clusters) + " clusters.")
    km = KMeans(n_clusters=num_clusters, verbose=1)

    km.fit(tfidf_matrix)
    # pickle km here.
    clusters = km.labels_.tolist()
    return km, clusters

from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


tfidf_vectorizer = TfidfVectorizer(max_df=0.80, max_features=200000,
                                 min_df=0.15, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


def get_tfidf_terms(list_X):
    print ('Running tfid...')
    tfidf_matrix = tfidf_vectorizer.fit_transform(list_X)
    print ('.. done')
    print ('Total vocab terms:' + str(len(tfidf_vectorizer.get_feature_names())))
    return {
        'tfidf_matrix' : tfidf_matrix ,
        'terms' : tfidf_vectorizer.get_feature_names()
    }
