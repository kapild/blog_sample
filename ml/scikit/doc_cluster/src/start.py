import pickle
import wikipedia
from bs4 import BeautifulSoup
import re
from HTMLParser import HTMLParser
from src.dump import dump_to_file, create_directory_if_not
import urllib2
from src.pickle_utils import load_pickle_or_run_and_save_function_pickle

__max_movies__ = 100
__version__ = "/new_wiki_search/"


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


    # for p in soup.findAll('ul', {'class': 'zebraList'}):
    #     for plot in p.findAll('p', {'class': 'plotSummary'}):

def get_imdb_movie_genres(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'link' in movie:
            imdb_link = movie["link"]
            genres = search_imbdb_movie_genre(imdb_link, index)
            movie['imdb_genres'] = genres

    return top_movies

def get_film_title_year(movie):
    return get_film_title_year(movie['title_raw'], movie['year'])

def get_film_title_year(title, year):
    return title + "_(" +year + "_film)"

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
                    'year' : year_type,
                    'title_year': get_film_title_year(a.text, year_type),
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


lambda_sort_wiki_title_by_film = lambda title: 'film' in title.lower()

def sort_wiki_titles(titles, year):
     titles = sorted(titles, key=lambda_sort_wiki_title_by_film, reverse=True)
     for index in range(0, len(titles)):
         title = titles[index]
         if year in title:
             return title
     return titles[0]

def remove_punctuations(input):
    return re.sub('[^A-Za-z0-9]+', ' ', input).lstrip().rstrip()

def search_wiki_url(film, year, index):
    url = "INVALID_PAGE"
    try:
        query = film
        query = unicode(query).encode('utf-8')
        print str(index) + ' querying movie:' + ":" + query
        wp_search = wikipedia.search(query)
        if len(wp_search) > 0:
            url = wikipedia.page(wp_search[0]).url
    except wikipedia.exceptions.DisambiguationError as e:

        url = INVALID_PAGE
        options_list = e.options
        import json
        # print 'Debug list:' + str(json.dumps(options_list, sort_keys=True, indent=4, separators=(',', ': ')))
        print 'sorting the list..'
        best_title = sort_wiki_titles(options_list, year)
        print 'best title:' + best_title
        best_title_url = _get_wiki_page_error(best_title)
        if best_title_url:
            url = best_title_url
        len_query = len(options_list)
        index = 0
        while index < len_query and url == INVALID_PAGE:
            opt = options_list[index]
            print "Debug film:" + opt
            index += 1
            if 'film' in opt or re.sub('[^A-Za-z0-9]+', ' ', opt) == film:
                print "query Debug film:" + opt
                wiki_url = _get_wiki_page_error(opt)
                if wiki_url:
                    url = wiki_url
    except Exception as e:
        print e
        url = INVALID_PAGE

    print 'url returned:' + url
    print
    return url


def _get_wiki_page_error(file_query):
    try:
        wp = wikipedia.page(file_query)
        return wp.url
    except wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError:
        return None


def search_and_add_wiki_url_links(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        query = movie['title']
        query += " movie film"
        year = ''
        if 'year' in movie:
            year = movie['year']
        query += " " + year
        url = search_wiki_url(query, year, index)
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
        synopses_imdb += div.text + " "
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


def get_wiki_synopsis_via_title(title_raw, try_with_new_title, message):
    print "Wiki raw title: " + title_raw + ", trying with " + message + "  title:" + try_with_new_title
    link = "http://en.wikipedia.org/wiki/" + try_with_new_title
    print "wiki link:" + link
    inner_synopses = search_wiki_synoposis(link)
    print inner_synopses
    return inner_synopses


def add_wiki_movies_synopsis_from_title_text(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'title_raw' in movie and movie['title_raw'] != "":
            title_raw = movie['title_raw']

            # underscore
            title_raw_with_underscore = title_raw.replace(" ", "_")
            movie['wiki_synopsis_title'] = get_wiki_synopsis_via_title(title_raw, title_raw_with_underscore,
                                                                       " underscore ")

            # _(film)
            title_raw_with_film = title_raw.replace(" ", "_") + "_(film)"
            movie['wiki_synopsis_title_film'] = get_wiki_synopsis_via_title(title_raw, title_raw_with_film,
                                                                            " film name")

            # (title_year)
            title_raw_year = movie['title_year'].replace(" ", "_")
            inner_synopses = get_wiki_synopsis_via_title(title_raw, title_raw_year, "_year_")
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
out_directory = __data_path + __version__
movie_pickle_path  = out_directory + "top_100_bolly.pkl"
movie_pickle_path_post_synopsuis = out_directory +  "top_100_bolly_synopsis.pkl"
movie_pickle_path_post_wiki_title_synopsis = out_directory + "top_100_bolly_wiki_title_synopsis.pkl"

movie_pickle_path_post_imdb_synopsis = out_directory + "top_100_bolly_imdb_synopsis.pkl"
movie_pickle_path_post_movie_genres = out_directory + "top_100_bolly_imdb_genres.pkl"
movie_final_scrape_path = out_directory + "movie_scrape.pkl"

def load_final_pickle_scrape():
    print "Loading final movie scrape using pickle:" + movie_final_scrape_path
    post_imdb_synopsis_top_movies = load_pickle_or_run_and_save_function_pickle(
        movie_final_scrape_path,
        " final movie scrape ",
        run_movie_scrape, None)
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


def load_and_save_imdb_movies():
    try:
        print "Loading pickle IMDB movies object:" + movie_pickle_path
        top_movies = pickle.load(open(movie_pickle_path, "rb"))
        print 'Loaded ' + str(len(top_movies)) + ' movies'
    except Exception as e:
        print "Pickle object not found at :" + movie_pickle_path
        top_movies = get_top_film_links_titles()
        search_and_add_wiki_url_links(top_movies)
        print "Pickling movie object:" + movie_pickle_path
        pickle.dump(top_movies, open(movie_pickle_path, "wb"))
    return top_movies


def run_movie_scrape(x):
    create_directory_if_not(out_directory)

    top_movies = load_and_save_imdb_movies()
    post_synopsis_top_movies = load_and_save_movie_wiki_synopsis(top_movies)
    post_imdb_synopsis_top_movies = load_and_save_imdb_movie_synopsis(post_synopsis_top_movies)
    post_genres = load_and_save_movie_genres(post_imdb_synopsis_top_movies)
    post_wikik_title_synopsis_top_movies = load_and_save_movie_wiki_from_title_synopsis(post_genres)

    dump_to_file(post_wikik_title_synopsis_top_movies, __version__, movie_final_scrape_path)

    print
    print "done"
    return post_wikik_title_synopsis_top_movies

if __name__ == "__main__":
    run_movie_scrape("")