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

import google
g = google.doGoogleSearch('Titanic film wikipedia')
g.pages = 5
print '*Found %s results*'%(g.get_result_count())
g.get_urls()