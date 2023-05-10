import pickle
 
with open('train_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(content.length)
with open('val_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(content.length)
with open('test_1.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(content.length)
