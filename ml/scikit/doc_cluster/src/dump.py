import json

list_of_f = [
    {'key': 'title', 'path': "titles"},
    {'key': 'title_year', 'path': "title_year_lists"},
    {'key': 'link','path': "imdb_links"},
    {'key': 'imdb_synopsis','path': "imdb_synopsis"},
    {'key': 'wiki_synopsis','path': "wiki_synopsis"},
    {'key': 'imdb_genres','path': "imdb_genres_lists"},
    {'key': 'wiki_synopsis_title','path': "wiki_synopsis_title_lists"},
    {'key': 'wiki_synopsis_title_year','path': "wiki_synopsis_title_lists"},
]

json_path = "../data/"  + "films.json"

def get_file_handler(list_of_f, __verison__= "_1"):
    file_handlers = dict()
    for d in list_of_f:
        file_handlers[d["key"]] = open("../data/" + d["path"] + __verison__ + ".txt", "wb")
    return file_handlers

def close_files(fhs):
    for key in fhs:
        fhs[key].close()

def dump_to_file(post_imdb_synopsis_top_movies, version):

    total_movies = len(post_imdb_synopsis_top_movies)
    fhs = get_file_handler(list_of_f, version)
    for index in range(0, total_movies):
        movie = post_imdb_synopsis_top_movies[index]
        for key in fhs:
            fh = fhs[key]
            line = movie.get(key, "")
            dump_line = json.dumps(line)
            # print "writing line:" + dump_line
            fh.write(dump_line.encode('utf-8').strip() + "\n")

    close_files(fhs)
    with open(json_path + version, 'w') as fp:
        fp.write(json.dumps(post_imdb_synopsis_top_movies, sort_keys=True, indent=4, separators=(',', ': ')))
