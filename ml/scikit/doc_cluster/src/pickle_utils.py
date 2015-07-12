import pickle
import math
def load_pickle_or_run_and_save_function_pickle(file_path, pickle_message, lambda_function, arg):
    try:
        print "Loading pickle " + pickle_message + " from path:" + file_path
        pickle_dict = pickle.load(open(file_path, "rb"))
        print 'Loaded objects.'
    except Exception as e:
        print "Not found, running:" + pickle_message + " by running lamda function:" + str(lambda_function)
        pickle_dict = lambda_function(arg)
        print "Pickling " + pickle_message + " " + file_path
        pickle.dump(pickle_dict, open(file_path, "wb"))
        print ".. done"

    return pickle_dict


def prit_get_sq(x):
    print "In lamdta:" + str (x)
    return math.sqrt(x)

square_root = lambda x: prit_get_sq(x)

if __name__ == "__main__":
    path = "test_pickle1"
    message = "new messge"
    x = load_pickle_or_run_and_save_function_pickle(path, message, square_root, 25)
    print str(x)

