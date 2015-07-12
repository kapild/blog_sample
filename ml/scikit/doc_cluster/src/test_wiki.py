import simplejson
import wikipedia
import re
INVALID_PAGE = "_INVALID_PAGE_URL"

def search_wiki_url(film, index):
    try:
        query = film + ' Film'
        query = unicode(query).encode('utf-8')
        print str(index) + ' querying movie:' + ":" + query
        wp = wikipedia.page(query)
        url = wp.url
    except wikipedia.exceptions.DisambiguationError as e:
        options_list = e.options
        import json
        print 'Debug list:' + str(json.dumps(options_list, sort_keys=True, indent=4, separators=(',', ': ')))
        url = INVALID_PAGE
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

def test():
    query = "Manorama 1959 film"
    search_wiki_url(query, 1)

def remove_punctuations(input):
    return re.sub('[^A-Za-z0-9]+', ' ', input).lstrip().rstrip()

if __name__ == "__main__":
    # print remove_punctuations(":3 Idiots_(2009_film)..")
    x = "\nNEW_LINE"
    print re.sub('NEW_LINE', ' ', x).lstrip().rstrip()
