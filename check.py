import pickle
 
with open('train_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
with open('val_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
with open('test_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(len(content))
