import urllib2
from bs4 import BeautifulSoup

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
import urllib
import simplejson
for i in titles[86:100]:
    print i
    query = urllib.urlencode({'q' : i + ' Film' + ' Wikipedia'})
    url = 'https://ajax.googleapis.com/ajax/services/search/web?v=1.0&' + query
    search_results = urllib2.urlopen(url)
    json = simplejson.loads(search_results.read())
    results = json['responseData']['results']
    links_wiki.append(results[0]['url'])
    print results[0]['url']
    print

titles

https://github.com/brandomr/document_cluster/blob/master/Film%20Scrape.ipynb