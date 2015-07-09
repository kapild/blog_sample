import urllib2
import wikipedia
from bs4 import BeautifulSoup
import re
from HTMLParser import HTMLParser

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
                title = re.sub('[^A-Za-z0-9]+', ' ', a.text)
                movie = {
                    'title': title,
                    'link': a['href']
                }
                movies.append(movie)
                print "Found " + str(index) + ": " + movie['title']
                index+=1

    print 'you collected ' + str(len(movies)) + ' movies link and titles.'
    print
    return movies


def add_wiki_links(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'title' in movie:
            title = movie['title']
            query = title + ' Film'
            query = unicode(query).encode('utf-8')
            print 'querying movie:' + str(index) + ":" + query
            url = ""
            try:
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
            movie['url'] = url


import urllib2
import simplejson

# The request also includes the userip parameter which provides the end
# user's IP address. Doing so will help distinguish this legitimate
# server-side traffic from traffic which doesn't come from an end-user.
# url = ('https://ajax.googleapis.com/ajax/services/search/web'
#        '?v=1.0&q=machine%20learning&userip=USERS-IP-ADDRESS')

# results = simplejson.load(response)
#
# request = urllib2.Request(
#     url, None, {'Referer': {}})
# response = urllib2.urlopen(request)
# results = simplejson.load(response)
# print '*Found %s results*'%(len(results['responseData']['results']))




# synopses_wiki_plot = []
#

def add_movies_synopsis(top_movies):
    total_movies = len(top_movies)
    for index in range(0, total_movies):
        movie = top_movies[index]
        if 'url' in movie and movie['url'] != INVALID_PAGE:
            link = movie['url']
            print "URL:" + link

            inner_synopses = ''
            request = urllib2.Request(link)
            response = urllib2.urlopen(request)
            soup = BeautifulSoup(response, "html.parser")
            patterns = ['Plot', 'Plot_summary', 'Plot_synopsis', 'Synopsis', 'Story']
            for pattern in patterns:
                print pattern
                if soup.find('span', {'id': pattern}):
                    next = soup.find('span', {'id': pattern}).next
                    break
            while next.name != "h2":
                newnext = ''
                print strip_tags(unicode(newnext).encode('utf-8','ignore'))
                try:
                    inner_synopses = inner_synopses + ' ' + strip_tags(unicode(next).encode('utf-8', 'ignore')) + ' '
                except:
                    innter_synopses = ''
                next = next.next
            print inner_synopses
            movie['synopsis'] = inner_synopses

# synopses_wiki = []
# for i in titles:
#     print "http://en.wikipedia.org/wiki/" + i.replace(' ','_')
#     request = urllib2.Request("http://en.wikipedia.org/wiki/" + i.replace(' ','_'))
#     response = urllib2.urlopen(request)
#     soup = BeautifulSoup(response, "html.parser")
#
#     inner_synopses = ''
#
#     for p in soup.findAll('p'):
#         print p.text
#         inner_synopses = inner_synopses + ' ' + p.text + ' '
#
#     synopses_wiki.append(inner_synopses)


# synopses_imdb = []
# for i in links:
#     print "http://www.imdb.com" + str(i) + "synopsis?ref_=tt_stry_pl"
#     request = urllib2.Request("http://www.imdb.com" + str(i) + "synopsis?ref_=tt_stry_pl")
#     response = urllib2.urlopen(request)
#     soup = BeautifulSoup(response, "html.parser")
#
#     for div in soup.findAll('div', {'id': 'swiki.2.1'}):
#         print div.text
#         synopses_imdb.append(div.text)
#
# genres = []
#
# for i in links:
#     print "http://www.imdb.com" + str(i)
#     request = urllib2.Request("http://www.imdb.com" + str(i))
#     response = urllib2.urlopen(request)
#     soup = BeautifulSoup(response, "html.parser")
#     for div in soup.findAll('div', {'itemprop': 'genre'}):
#         genres_inner = []
#         for a in div.findAll('a'):
#             genres_inner.append(a.text)
#         genres.append(genres_inner)
#         print genres_inner
#         print
#
# len(genres)
#
# title_list = open('title_list.txt', 'w')
#
# for item in titles:
#   title_list.write("%s\n" % item)
#
# title_list.close()

#wiki links
# links_list = open('link_list_wiki.txt', 'w')
# for item in links_wiki_new:
#   links_list.write("%s\n" % item)
# links_list.close()




# for i in range(len(links)):
#     links[i] = 'http://www.imdb.com' + str(links[i])

#imdb links
# links_list = open('link_list_imdb.txt', 'w')
#
# links_imdb = links
# for item in links_imdb:
#   links_list.write("%s\n" % item)
# links_list.close()
#
# #wiki synopses
# synopses_list = open('synopses_list_wiki.txt', 'w')
# for item in synopses_wiki_plot:
#   synopses_list.write("%s\n BREAKS HERE" % item)
# synopses_list.close()
#
# #imdb synopses
# synopses_list = open('synopses_list_imdb.txt', 'w')
# for item in synopses_imdb:
#   synopses_list.write("%s\n BREAKS HERE" % item.encode('utf-8'))
# synopses_list.close()
#
# genres_list = open('genres_list.txt', 'w')
# for item in genres:
#   genres_list.write("%s\n" % item)
#
# genres_list.close()
# # import urllib2
# # opener = urllib2.build_opener()
# # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# # infile = opener.open('http://en.wikipedia.org/w/index.php?title=Albert_Einstein&printable=yes')
# # page = infile.read()


# https://github.com/brandomr/document_cluster/blob/master/Film%20Scrape.ipynb

movie_pickle_path  = "top_100_bolly.pkl"
import pickle
def run_movie_scrape():

    try:
        print "Loading pickle movie object:" + movie_pickle_path
        top_movies = pickle.load( open(movie_pickle_path, "rb"))
        print 'Loaded ' + str(len(top_movies)) + ' movies'
    except Exception as e:
        print "Pickle object not found at :" + movie_pickle_path
        top_movies = load_and_save_movie_pickle()

    add_movies_synopsis(top_movies)

def load_and_save_movie_pickle():
    try:
        top_movies = get_top_film_links_titles()
        add_wiki_links(top_movies)
        print "Pickling movie object:" + movie_pickle_path
    except Exception as e:
        print "Pickling part movie object:" + movie_pickle_path
        pickle.dump(top_movies, open(movie_pickle_path, "wb"))
    return top_movies


if __name__ == "__main__":
    run_movie_scrape()