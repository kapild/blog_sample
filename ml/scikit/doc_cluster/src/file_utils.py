def read_file_to_list(file_path):
    fh = open(file_path + ".txt", "r")
    list_X  = []
    for line in fh:
        list_X.append(line)
    print "Read " + str(len(list_X)) + " lines from path:" + file_path
    fh.close()
    return list_X
