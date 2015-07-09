import simplejson
import wikipedia


def test():
    query = "Chakde! India Film"
    print 'query:' + query
    try:
        wp = wikipedia.page(query)
    except wikipedia.exceptions.DisambiguationError as e:
        options_list = e.options
        for opt in options_list:
            if 'film' in opt:
                wp = wikipedia.page(opt)
                break
    except wikipedia.exceptions.PageError as e:
        print e



if __name__ == "__main__":
    test()
