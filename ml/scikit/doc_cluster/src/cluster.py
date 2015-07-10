from __future__ import print_function as printk
from bs4 import BeautifulSoup
import re
import pandas as pd
from src.start import load_final_pickle_scrape
import nltk

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

list_clean_text =   ['imdb_synopsis',
                     'wiki_synopsis',
                     'wiki_synopsis_title',
                     'wiki_synopsis_title_year'
]

from sklearn.feature_extraction.text import TfidfVectorizer


COMBINED_SYNOPSIS = "combibed_synopsis"
TOKENS_STEMS = "tokens_stemmed"
TOKENS_UNSTEMMED = "tokens_only"

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


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


def init_clean_combine_scrape(top_movies):
    print ('Cleaning and combining synopsis.')
    total_count = len(top_movies)
    for index in range(0, total_count):
        movie = top_movies[index]
        for key_ in list_clean_text:
            if key_ in movie:
                movie[key_] = clean_wiki_text(movie[key_])
            else:
                movie[key_]  = ""
        # get combined synopsis
        movie[COMBINED_SYNOPSIS] = get_combined_synopsis_text(movie)

def get_combined_synopsis_text(movie):
    synopsis_combined = ""
    for key_ in list_clean_text:
        synopsis = movie[key_]
        synopsis_combined += synopsis + " "
    return synopsis_combined

def clean_wiki_text(raw_text):
    text = BeautifulSoup(raw_text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    # if len(raw_text) != len(text):
        # print "Striped:" + text
    return text


def get_k_means_cluster(tfidf_matrix, num_clusters):
    from sklearn.cluster import KMeans

    print ("Running k-means on " + str(num_clusters) + " clusters.")
    km = KMeans(n_clusters=num_clusters)

    km.fit(tfidf_matrix)
    # pickle km here.
    clusters = km.labels_.tolist()
    return km, clusters


def get_clusters_in_frame(clusters, top_movies):
    titles = []
    synopses = []
    genres = []
    ranks = []

    for i in range(0,len(top_movies)):
        ranks.append(i)
        movie = top_movies[i]
        titles.append(movie['title'])
        synopses.append(movie[COMBINED_SYNOPSIS])
        genres.append(movie['imdb_genres'])

    films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }

    frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])
    print ("Frame cluster count:\n" + str(frame['cluster'].value_counts()))
    print
    grouped = frame['rank'].groupby(frame['cluster'])
    print ("Group mean:\n" + str(grouped.mean()))
    print ()
    return frame

def print_top_k_terms_cluster(km, terms, vocab_frame, frame, num_clusters, top_k=10):
    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :top_k]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print()
        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print()
        print()

def get_synopsis_stems(top_movies):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    synopysis_list = []
    total_count = len(top_movies)
    print ("Storing stemmed and un-stemmed synopsis.")
    for index in range(0, total_count):
        movie = top_movies[index]
        synopysis = movie[COMBINED_SYNOPSIS]
        synopysis_list.append(synopysis)
        movie[TOKENS_STEMS] = tokenize_and_stem(synopysis)
        totalvocab_stemmed.extend(movie[TOKENS_STEMS])
        movie[TOKENS_UNSTEMMED] = tokenize_only(synopysis)
        totalvocab_tokenized.extend(movie[TOKENS_UNSTEMMED])

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

    print ('Running tfid...')
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopysis_list)
    print ('.. done')
    terms = tfidf_vectorizer.get_feature_names()

    print ("Shape tfidf Matrix: " + str(tfidf_matrix.shape))
    return vocab_frame, tfidf_matrix, terms


    print
if __name__ == "__main__":
    top_movies = load_final_pickle_scrape()
    init_clean_combine_scrape(top_movies)
    vocab_frame, tfidf_matrix, terms = get_synopsis_stems(top_movies)
    num_clusters = 7
    km, clusters = get_k_means_cluster(tfidf_matrix, num_clusters)
    frame = get_clusters_in_frame(clusters, top_movies)
    print_top_k_terms_cluster(km, terms, vocab_frame, frame, num_clusters)
