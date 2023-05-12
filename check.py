import pickle
 
with open('train_2.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
with open('val_2.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
with open('test_2.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
