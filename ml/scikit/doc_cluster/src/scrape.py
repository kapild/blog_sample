#
# # import urllib2
# # opener = urllib2.build_opener()
# # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# # infile = opener.open('http://en.wikipedia.org/w/index.php?title=Albert_Einstein&printable=yes')
# # page = infile.read()


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




