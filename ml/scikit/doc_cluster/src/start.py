import pickle
import wikipedia
from bs4 import BeautifulSoup
import re
from HTMLParser import HTMLParser
from src.dump import dump_to_file
import urllib2
__max_movies__ = 100
__version__ = "_100_movies"


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

INVALID_PAGE = "_INVALID_PAGE_URL"
top_hindi_link = "http://www.imdb.com/list/ls006090789/?start=1&view=detail&sort=user_rating:desc&defaults=1&scb=0.02446836233139038";


def search_imbdb_movie_genre(imdb_link, index):
    print "Querying links for Genres:"  + str(index) + ", " + imdb_link
    request = urllib2.Request(imdb_link)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, "html.parser")
    genres_inner = []

    for div in soup.findAll('div', {'itemprop': 'genre'}):
        for a in div.findAll('a'):
            genres_inner.append(a.text)
    print genres_inner
    print
    return genres_inner


def get_imdb_movie_genres(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'link' in movie:
            imdb_link = movie["link"]
            genres = search_imbdb_movie_genre(imdb_link, index)
            movie['imdb_genres'] = genres

    return top_movies

def get_top_film_links_titles():
    request = urllib2.Request(top_hindi_link)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, "html.parser")
    type(soup)
    # for div in soup.findAll('div', {'class': 'info'}):
    #     for b in soup.findAll('b'):
    #         for a in b.findAll('a'):
    #             # print a.text
    movies = []
    index = 1
    for div in soup.findAll('div', {'class': 'info'}):
        for b in div.findAll('b'):
            for a in b.findAll('a'):
                spans = b.findAll('span')
                year_type = ""
                if spans and len(spans) > 0:
                    year_type = re.sub('[()]+', '', spans[0].text)
                title = re.sub('[^A-Za-z0-9]+', ' ', a.text)
                movie = {
                    'title': title,
                    'title_raw' : a.text,
                    'title_year': a.text + "_(" + year_type + "_film)",
                    # 'title_year': a.text + "_(" + "_film)",
                    'link': "http://www.imdb.com/" + a['href']
                }
                movies.append(movie)
                print "Found " + str(index) + ": " + movie['title']
                if index>= __max_movies__ :
                    print 'Returning max movies: ' + str(index)
                    return movies
                index+=1

    print 'you collected ' + str(len(movies)) + ' movies link and titles.'
    print
    return movies


def search_wiki_url(query, index):
    try:
        query += ' Film'
        query = unicode(query).encode('utf-8')
        print 'querying movie:'
        print str(index) + ":" + query
        wp = wikipedia.page(query)
        url = wp.url
    except wikipedia.exceptions.DisambiguationError as e:
        options_list = e.options
        for opt in options_list:
            if 'film' in opt:
                wp = wikipedia.page(opt)
                url = wp.url
                break
    except wikipedia.exceptions.PageError as e:
        url = INVALID_PAGE
    print 'url returned:' + url
    print
    return url


def add_wiki_links(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'title' in movie:
            title = movie['title']
            url = search_wiki_url(title, index)
            movie['url'] = url


def search_imdb_synopsis(imdb_link_ref_link):
    imdb_link = imdb_link_ref_link + "synopsis?ref_=tt_stry_pl"
    print "Imdb link:" + imdb_link
    request = urllib2.Request(imdb_link)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, "html.parser")
    synopses_imdb = ""
    for div in soup.findAll('div', {'id': 'swiki.2.1'}):
        print div.text
        synopses_imdb += div.text + "NEW_LINE"
    return synopses_imdb

def add_imdb_synopsis(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'link' in movie and (movie['link'] != "" and movie['link'] != INVALID_PAGE):
            link = movie['link']
            print "Querying imdb: " + str(index) + " " + link
            inner_synopses = search_imdb_synopsis(link)
            print str(index) + " imdb synopsis:" + inner_synopses
            movie['imdb_synopsis'] = inner_synopses
    return top_movies


def get_response(link):
    try:
        request = urllib2.Request(link)
        return urllib2.urlopen(request)
    except urllib2.HTTPError, urllib2   .URLError:
        print "Error while getting response from link:" + link
        return None


def search_wiki_synoposis(link):
    print "URL:" + link
    inner_synopses = ''
    response = get_response(link)
    if response:
        soup = BeautifulSoup(response, "html.parser")
        patterns = ['Plot', 'Plot_summary', 'Plot_synopsis', 'Synopsis', 'Story']
        next = None
        for pattern in patterns:
            print pattern
            if soup.find('span', {'id': pattern}):
                next = soup.find('span', {'id': pattern}).next
                break

        if next:
            while next.name != "h2":
                newnext = ''
                print strip_tags(unicode(newnext).encode('utf-8', 'ignore'))
                try:
                    inner_synopses = inner_synopses + ' ' + strip_tags(unicode(next).encode('utf-8', 'ignore')) + ' '
                except:
                    pass
                next = next.next
        return inner_synopses
    return ""

def add_wiki_movies_synopsis_from_title_text(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'title_raw' in movie and movie['title_raw'] != "":
            title_raw = movie['title_raw']
            title_raw_with_underscore = title_raw.replace(" ", "_")
            print "Wiki raw title: " +  title_raw + " _Underscore:" + title_raw_with_underscore
            link = "http://en.wikipedia.org/wiki/" + title_raw_with_underscore
            print "wiki link:" + link
            inner_synopses = search_wiki_synoposis(link)
            print inner_synopses
            movie['wiki_synopsis_title'] = inner_synopses
            if 'title_year' in movie and movie['title_year'] != "":
                title_raw_year = movie['title_year'].replace(" ", "_")
                link = "http://en.wikipedia.org/wiki/" + title_raw_year
                print "wiki (with year) link:" + link
                inner_synopses = search_wiki_synoposis(link)
                print inner_synopses
                movie['wiki_synopsis_title_year'] = inner_synopses



def add_wiki_movies_synopsis_from_url(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'url' in movie and (movie['url'] != "" and movie['url'] != INVALID_PAGE):
            link = movie['url']
            inner_synopses = search_wiki_synoposis(link)
            print inner_synopses
            movie['wiki_synopsis'] = inner_synopses

__data_path = "../data/"
movie_pickle_path  = __data_path + "top_100_bolly.pkl" + __version__
movie_pickle_path_post_synopsuis = __data_path + "top_100_bolly_synopsis.pkl"  + __version__
movie_pickle_path_post_wiki_title_synopsis = __data_path + "top_100_bolly_wiki_title_synopsis.pkl"  + __version__

movie_pickle_path_post_imdb_synopsis = __data_path + "top_100_bolly_imdb_synopsis.pkl" +  __version__
movie_pickle_path_post_movie_genres = __data_path + "top_100_bolly_imdb_genres.pkl" +  __version__

movie_final_scrape_path = __data_path + "movie_scrape" + __version__ + ".pkl"


def load_final_pickle_scrape():
    print "Loading final movie scrape using pickle:" + movie_final_scrape_path
    post_imdb_synopsis_top_movies = pickle.load(open(movie_final_scrape_path, "rb"))
    print 'Loaded ' + str(len(post_imdb_synopsis_top_movies)) + ' movies'
    return post_imdb_synopsis_top_movies


def load_and_save_imdb_movie_synopsis(post_synopsis_top_movies):
    try:
        print "Loading pickle movie object(post imdb synopsis):" + movie_pickle_path_post_imdb_synopsis
        post_imdb_synopsis_top_movies = pickle.load(open(movie_pickle_path_post_imdb_synopsis, "rb"))
        print 'Loaded ' + str(len(post_imdb_synopsis_top_movies)) + ' movies'
    except Exception as e:
        print "Getting synopsis."
        post_imdb_synopsis_top_movies = add_imdb_synopsis(post_synopsis_top_movies)
        pickle.dump(post_imdb_synopsis_top_movies, open(movie_pickle_path_post_imdb_synopsis, "wb"))
        print "Pickling movie objects (post imdb synopsis):" + movie_pickle_path_post_imdb_synopsis

    return post_imdb_synopsis_top_movies

def load_and_save_movie_genres(post_synopsis_top_movies):
    try:
        print "Loading pickle movie object(post genres):" + movie_pickle_path_post_movie_genres
        post_imdb_genres_top_movies = pickle.load(open(movie_pickle_path_post_movie_genres, "rb"))
        print 'Loaded genres:' + str(len(post_imdb_genres_top_movies)) + ' movies'
    except Exception as e:
        print "Getting genres:"
        post_imdb_genres_top_movies = get_imdb_movie_genres(post_synopsis_top_movies)
        pickle.dump(post_imdb_genres_top_movies, open(movie_pickle_path_post_movie_genres, "wb"))
        print "Pickling movie objects (post post genres):" + movie_pickle_path_post_movie_genres
    return post_imdb_genres_top_movies


def load_and_save_movie_wiki_synopsis(top_movies):
    try:
        print "Loading pickle movie object(post wiki synopsis):" + movie_pickle_path_post_synopsuis
        post_synopsis_top_movies = pickle.load(open(movie_pickle_path_post_synopsuis, "rb"))
        print 'Loaded wiki synopsis' + str(len(post_synopsis_top_movies)) + ' movies'
        return post_synopsis_top_movies
    except Exception as e:
        print "Getting wiki synopsis."
        add_wiki_movies_synopsis_from_url(top_movies)
        pickle.dump(top_movies, open(movie_pickle_path_post_synopsuis, "wb"))
        print "Pickling movie objects (post wiki synopsis):" + movie_pickle_path_post_synopsuis

        return top_movies

def load_and_save_movie_wiki_from_title_synopsis(top_movies):
    try:
        print "Loading pickle movie object(post wiki title synopsis):" + movie_pickle_path_post_wiki_title_synopsis
        post_synopsis_top_movies = pickle.load(open(movie_pickle_path_post_wiki_title_synopsis, "rb"))
        print 'Loaded wiki synopsis' + str(len(post_synopsis_top_movies)) + ' movies'
        return post_synopsis_top_movies
    except Exception as e:
        print "Getting wiki (from title) synopsis."
        add_wiki_movies_synopsis_from_title_text(top_movies)
        print "Pickling movie objects (post wiki title synopsis):" + movie_pickle_path_post_wiki_title_synopsis
        pickle.dump(top_movies, open(movie_pickle_path_post_wiki_title_synopsis, "wb"))
        return top_movies


def load_and_save_movie_pickle():
    try:
        top_movies = get_top_film_links_titles()
        add_wiki_links(top_movies)
        print "Pickling movie object:" + movie_pickle_path
        pickle.dump(top_movies, open(movie_pickle_path, "wb"))
    except Exception as e:
        print "Pickling part movie object:" + movie_pickle_path
        pickle.dump(top_movies, open(movie_pickle_path, "wb"))
    return top_movies


def run_movie_scrape():

    try:
        print "Loading pickle movie object:" + movie_pickle_path
        top_movies = pickle.load(open(movie_pickle_path, "rb"))
        print 'Loaded ' + str(len(top_movies)) + ' movies'
    except Exception as e:
        print "Pickle object not found at :" + movie_pickle_path
        top_movies = load_and_save_movie_pickle()

    post_synopsis_top_movies = load_and_save_movie_wiki_synopsis(top_movies)
    post_imdb_synopsis_top_movies = load_and_save_imdb_movie_synopsis(post_synopsis_top_movies)
    post_genres = load_and_save_movie_genres(post_imdb_synopsis_top_movies)
    post_wikik_title_synopsis_top_movies = load_and_save_movie_wiki_from_title_synopsis(post_genres)

    dump_to_file(post_wikik_title_synopsis_top_movies, __version__, movie_final_scrape_path)

    print
    print "done"

if __name__ == "__main__":
    run_movie_scrape()