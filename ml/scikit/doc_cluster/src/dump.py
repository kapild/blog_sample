import json
import pickle
import os

movie_file_lists = [
    {'key': 'title', 'path': "titles"},
    {'key': 'title_year', 'path': "title_year_lists"},
    {'key': 'link','path': "imdb_links"},
    {'key': 'imdb_synopsis','path': "imdb_synopsis"},
    {'key': 'wiki_synopsis','path': "wiki_synopsis"},
    {'key': 'imdb_genres','path': "imdb_genres_lists"},
    {'key': 'wiki_synopsis_title','path': "wiki_synopsis_title_lists"},
    {'key': 'wiki_synopsis_title_year','path': "wiki_synopsis_title_year_lists"},
    {'key': 'wiki_synopsis_title_film','path': "wiki_synopsis_title_film_lists"},


]


cluster_terms = [
    {'key': 'synossis_list', 'path': "synopsis_list"},
]
def get_file_handler(list_of_f, out_dir):
    file_handlers = dict()
    for d in list_of_f:
        file_handlers[d["key"]] = open(out_dir + d["path"] + ".txt", "wb")
    return file_handlers

def close_files(fhs):
    for key in fhs:
        fhs[key].close()

def dump_movie_pickle_data(top_movies, out_dir, movie_path):
    dump_to_file(top_movies, out_dir, movie_path, movie_file_lists)

def dump_cluster_terms_to_file(cluster_obj, terms_path):
    dump_to_file(cluster_obj, "", terms_path, cluster_terms)

def dump_to_file(dump_obj, out_dir, pickle_path, list_of_f):
    total_movies = len(dump_obj)
    fhs = get_file_handler(list_of_f, out_dir)
    for index in range(0, total_movies):
        movie = dump_obj[index]
        for key in fhs:
            fh = fhs[key]
            line = movie.get(key, "")
            dump_line = json.dumps(line)
            # print "writing line:" + dump_line
            fh.write(dump_line.encode('utf-8').strip() + "\n")
    close_files(fhs)

    json_path = out_dir + "films.json"
    print
    print "Writing movie data to json file:" + json_path
    with open(json_path, 'w') as fp:
        fp.write(json.dumps(dump_obj, sort_keys=True, indent=4, separators=(',', ': ')))

    print "..done"
    # pcikle the final scrape
    pickle.dump(dump_obj, open(pickle_path, "wb"))


def dump_list_to_file(dump_list, dump_path):
    total_movies = len(dump_list)
    fh = open(dump_path + ".txt", "wb")
    for index in range(0, total_movies):
        dump_line = dump_list[index]
        # print "writing line:" + dump_line
        fh.write(dump_line.encode('utf-8').strip() + "\n")
    fh.close()

def read_file_to_list(file_path):
    fh = open(file_path + ".txt", "r")
    list_X  = []
    for line in fh:
        list_X.append(line)
    print "Read " + str(len(list_X)) + " lines from path:" + file_path
    fh.close()
    return list_X

def create_directory_if_not(directory):
    if not os.path.exists(directory):
        print
        print "Creating a new directory at location:" + str(directory)
        print
        os.makedirs(directory)