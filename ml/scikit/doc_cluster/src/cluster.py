from __future__ import print_function as printk
from bs4 import BeautifulSoup
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.dump import create_directory_if_not, dump_cluster_terms_to_file, dump_list_to_file
from src.pickle_utils import load_pickle_or_run_and_save_function_pickle
from src.start import load_final_pickle_scrape, remove_punctuations, get_out_directory
import nltk
import matplotlib.pyplot as plt
import mpld3

from sklearn.manifold import MDS
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

list_clean_text =   ['imdb_synopsis',
                     'wiki_synopsis',
                     'wiki_synopsis_title',
                     'wiki_synopsis_title_year',
                     'wiki_synopsis_title_film'
]

from sklearn.feature_extraction.text import TfidfVectorizer


COMBINED_SYNOPSIS = "combibed_synopsis"
TOKENS_STEMS = "tokens_stemmed"
TOKENS_UNSTEMMED = "tokens_only"

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

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

tfidf_vectorizer = TfidfVectorizer(max_df=0.80, max_features=200000,
                                 min_df=0.15, stop_words='english',
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
                movie[key_] = ""
        # get combined synopsis
        movie[COMBINED_SYNOPSIS] = get_combined_synopsis_text(movie)

def get_combined_synopsis_text(movie):
    synopsis_combined = ""
    last_synopsis_inserted = None
    for key_ in list_clean_text:
        synopsis = movie[key_]
        if synopsis and synopsis != '':
            # if first time addition or not previously added.
            if last_synopsis_inserted == None or last_synopsis_inserted != synopsis:
                last_synopsis_inserted = synopsis
                synopsis_combined += synopsis + " "
    return synopsis_combined

def clean_wiki_text(raw_text):
    text = BeautifulSoup(raw_text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    # if len(raw_text) != len(text):
        # print "Striped:" + text

    text = re.sub('NEW_LINE', ' ', text).strip()
    pun_removed = remove_punctuations(text)
    return pun_removed


def get_k_means_cluster(tfidf_matrix, num_clusters, is_list=True):
    from sklearn.cluster import KMeans

    print ("Running k-means on " + str(num_clusters) + " clusters.")
    km = KMeans(n_clusters=num_clusters, verbose=0)

    km.fit(tfidf_matrix)
    # pickle km here.
    clusters = km.labels_
    if is_list:
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
    return frame, titles

def get_top_k_terms_cluster(km, terms, vocab_frame, frame, num_clusters, top_k=30):
    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    top_words_list = []
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')
        top_words = ""
        for ind in order_centroids[i, :top_k]:
            top_word = vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')
            top_words += top_word + ", "
            print(' %s' % top_word, end=',')
        top_words_list.append(top_words)
        print()
        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print()
        print()
    return top_words_list


def perform_mds(tfidf_matrix, titles, top_words_list, clusters, cluster_out_dir):
    MDS()
    dist = 1 - cosine_similarity(tfidf_matrix)
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()
    visualzie_doc_cluster(xs, ys, titles, top_words_list, clusters)
    # visualize_doc_cluster_d3(xs, ys, titles, top_words_list, clusters)
    hierarchical_clustering(dist, cluster_out_dir)

def visualize_doc_cluster_d3(xs, ys, titles, top_words_list, clusters):
    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    #group by cluster
    groups = df.groupby('label')
        #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {0: top_words_list[0],
                 1: top_words_list[1],
                 2: top_words_list[2],
                 3: top_words_list[3],
                 4: top_words_list[4]}

    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: -200px;}
    """

    # Plot
    fig, ax = plt.subplots(figsize=(14,6)) #set plot size
    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                         label=cluster_names[name], mec='none',
                         color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]

        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                           voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


    ax.legend(numpoints=1) #show legend with only one dot

    mpld3.display() #show the plot

    #uncomment the below to export to html
    html = mpld3.fig_to_html(fig)
    print(html)

def visualzie_doc_cluster(xs, ys, titles, top_words_list, clusters):
    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',
                      5: '#aaaaaa', 6: '#bbbbbb', 7: '#cccccc', 8: '#dddddd', 9: '#eeeeee'}

    #set up cluster names using a dict
    # cluster_names = {0: top_words_list[0],
    #              1: top_words_list[1],
    #              2: top_words_list[2],
    #              3: top_words_list[3],
    #              4: top_words_list[4]}

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    all_df = dict(x=xs, y=ys, label=clusters, title=titles)
    df = pd.DataFrame(all_df)
    #group by cluster
    groups = df.groupby('label')
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                # label=cluster_names[name],
                color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

    plt.show() #show the plot





lambda_stem_un_stemm_synopsis = lambda x: stem_un_stem_synopsis(x)

lambda_tfidf_terms = lambda x: get_tfidf_terms(x)


def get_tfidf_terms(list_X):
    print ('Running tfid...')
    tfidf_matrix = tfidf_vectorizer.fit_transform(list_X)
    print ('.. done')
    print ('Total vocab terms:' + str(len(tfidf_vectorizer.get_feature_names())))
    return {
        'tfidf_matrix' : tfidf_matrix ,
        'terms' : tfidf_vectorizer.get_feature_names()
    }

def stem_un_stem_synopsis(top_movies):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    synopysis_list = []
    total_count = len(top_movies)
    print("Performing stemmed and un-stemmed synopsis.")
    for index in range(0, total_count):
        movie = top_movies[index]
        synopysis = movie[COMBINED_SYNOPSIS]
        synopysis_list.append(synopysis)
        movie[TOKENS_STEMS] = tokenize_and_stem(synopysis)
        totalvocab_stemmed.extend(movie[TOKENS_STEMS])
        movie[TOKENS_UNSTEMMED] = tokenize_only(synopysis)
        totalvocab_tokenized.extend(movie[TOKENS_UNSTEMMED])

    pickle_obj = {
        "synossis_list": synopysis_list,
        "totalvocab_stemmed": totalvocab_stemmed,
        "totalvocab_tokenized": totalvocab_tokenized,
        "top_movies": top_movies
    }
    return pickle_obj


def hierarchical_clustering(dist, cluster_out_dir):
    from scipy.cluster.hierarchy import ward, dendrogram

    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout

    #uncomment below to save figure
    hireachical_cluster_path = cluster_out_dir + "cluster.png"

    plt.savefig(hireachical_cluster_path, dpi=200) #save figure as ward_clusters
    plt.close()


def get_tf_idf_matix_terms(cluster_out_dir, list_X):
    synopsis_tfidf_terms_path = cluster_out_dir + "tfidf_terms.pkl"
    tfidf_terms = load_pickle_or_run_and_save_function_pickle(
        synopsis_tfidf_terms_path,
        "Tf-idf and vocab terms.",
        lambda_tfidf_terms,
        list_X
        # TRY with STEM version also.
    )
    return tfidf_terms


def get_terms_path(cluster_out_dir):
    return cluster_out_dir + "/terms"


def get_synopsis_stems(top_movies, cluster_out_dir):

    synopsis_stems_unstemmed_path = cluster_out_dir + "_movie_stem_unstemm.pkl"

    stem_unstem = load_pickle_or_run_and_save_function_pickle(
        synopsis_stems_unstemmed_path,
        "Stemming and un-stemming synopsis.",
        lambda_stem_un_stemm_synopsis,
        top_movies
    )
    tfidf_terms = get_tf_idf_matix_terms(cluster_out_dir, stem_unstem['synossis_list'])
    dump_list_to_file(stem_unstem['synossis_list'], get_terms_path(cluster_out_dir))

    print ("Shape tfidf Matrix: " + str(tfidf_terms['tfidf_matrix'].shape))

    vocab_frame = pd.DataFrame({'words': stem_unstem['totalvocab_tokenized']}, index=stem_unstem['totalvocab_stemmed'])
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    print (vocab_frame.head())
    print ()
    print ()
    print ()
    print ()

    return vocab_frame, tfidf_terms['tfidf_matrix'], tfidf_terms['terms']



def run_clustering(cluster_out_dir):
    global top_movies, vocab_frame, tfidf_matrix, terms, num_clusters, km, clusters, frame, titles, top_words_list
    create_directory_if_not(cluster_out_dir)
    top_movies = load_final_pickle_scrape(get_out_directory())
    init_clean_combine_scrape(top_movies)
    vocab_frame, tfidf_matrix, terms = get_synopsis_stems(top_movies, cluster_out_dir)
    num_clusters = 6
    km, clusters = get_k_means_cluster(tfidf_matrix, num_clusters)
    frame, titles = get_clusters_in_frame(clusters, top_movies)
    top_words_list = get_top_k_terms_cluster(km, terms, vocab_frame, frame, num_clusters)
    perform_mds(tfidf_matrix, titles, top_words_list, clusters, cluster_out_dir)

_cluster_dir = "/cluser_2/"


def get_cluster_directory(out_directory):
    return out_directory + _cluster_dir


if __name__ == "__main__":
    out_directory = get_out_directory()
    run_clustering(get_cluster_directory(out_directory))