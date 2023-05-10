from ldcNormsCombine import generate_input
import pickle

train = generate_input('INTERNAL_TRAIN',1,1)
val =generate_input('INTERNAL_VAL',1,1)
test = generate_input('INTERNAL_TEST',1,1)



with open('val_1.pickle', 'wb') as f:
    # Write the list of objects to the file
    pickle.dump(val, f)

with open('test_1.pickle', 'wb') as f:
    # Write the list of objects to the file
    pickle.dump(test, f)