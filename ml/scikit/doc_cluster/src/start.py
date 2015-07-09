import urllib2
from bs4 import BeautifulSoup

INVALID_PAGE = "_INVALID_PAGE_URL"
top_hindi_link = "http://www.imdb.com/list/ls006090789/?start=1&view=detail&sort=user_rating:desc&defaults=1&scb=0.02446836233139038";

request = urllib2.Request(top_hindi_link)
response = urllib2.urlopen(request)
soup = BeautifulSoup(response, "html.parser")
type(soup)

for div in soup.findAll('div', {'class': 'info'}):
    for b in soup.findAll('b'):
        for a in b.findAll('a'):
            print a.text


links = []
titles = []

for div in soup.findAll('div', {'class': 'info'}):
    for b in div.findAll('b'):
        for a in b.findAll('a'):
            titles.append(a.text)
            links.append(a['href'])
            print a.text

print 'you collected ' + str(len(links)) + ' links.'
print
print 'you collected ' + str(len(titles)) + ' titles.'

str(links[0])


import urllib2
import simplejson

# The request also includes the userip parameter which provides the end
# user's IP address. Doing so will help distinguish this legitimate
# server-side traffic from traffic which doesn't come from an end-user.
url = ('https://ajax.googleapis.com/ajax/services/search/web'
       '?v=1.0&q=machine%20learning&userip=USERS-IP-ADDRESS')

results = simplejson.load(response)

request = urllib2.Request(
    url, None, {'Referer': {}})
response = urllib2.urlopen(request)
results = simplejson.load(response)
print '*Found %s results*'%(len(results['responseData']['results']))

links_wiki = []

len(titles)
import simplejson
import wikipedia

for i in titles[0:100]:
    query = i + ' Film'
    print 'query:' + query
    query = unicode(query).encode('utf-8')
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
    print url
    links_wiki.append(url)

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

synopses_wiki_plot = []

for i in links_wiki:
    print i
    link = i
    inner_synopses = ''
    if i != INVALID_PAGE:
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
    synopses_wiki_plot.append(inner_synopses)



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


synopses_imdb = []
for i in links:
    print "http://www.imdb.com" + str(i) + "synopsis?ref_=tt_stry_pl"
    request = urllib2.Request("http://www.imdb.com" + str(i) + "synopsis?ref_=tt_stry_pl")
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, "html.parser")

    for div in soup.findAll('div', {'id': 'swiki.2.1'}):
        print div.text
        synopses_imdb.append(div.text)

genres = []

for i in links:
    print "http://www.imdb.com" + str(i)
    request = urllib2.Request("http://www.imdb.com" + str(i))
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, "html.parser")
    for div in soup.findAll('div', {'itemprop': 'genre'}):
        genres_inner = []
        for a in div.findAll('a'):
            genres_inner.append(a.text)
        genres.append(genres_inner)
        print genres_inner
        print

len(genres)

title_list = open('title_list.txt', 'w')

for item in titles:
  title_list.write("%s\n" % item)

title_list.close()

#wiki links
# links_list = open('link_list_wiki.txt', 'w')
# for item in links_wiki_new:
#   links_list.write("%s\n" % item)
# links_list.close()




for i in range(len(links)):
    links[i] = 'http://www.imdb.com' + str(links[i])

#imdb links
links_list = open('link_list_imdb.txt', 'w')

links_imdb = links
for item in links_imdb:
  links_list.write("%s\n" % item)
links_list.close()

#wiki synopses
synopses_list = open('synopses_list_wiki.txt', 'w')
for item in synopses_wiki_plot:
  synopses_list.write("%s\n BREAKS HERE" % item)
synopses_list.close()

#imdb synopses
synopses_list = open('synopses_list_imdb.txt', 'w')
for item in synopses_imdb:
  synopses_list.write("%s\n BREAKS HERE" % item.encode('utf-8'))
synopses_list.close()

genres_list = open('genres_list.txt', 'w')
for item in genres:
  genres_list.write("%s\n" % item)

genres_list.close()
# import urllib2
# opener = urllib2.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# infile = opener.open('http://en.wikipedia.org/w/index.php?title=Albert_Einstein&printable=yes')
# page = infile.read()

titles

# https://github.com/brandomr/document_cluster/blob/master/Film%20Scrape.ipynb