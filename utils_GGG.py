from keras.utils import to_categorical
import numpy as np


def padding(mat, max_seq):
    # =============================================================================
    #     This function creates zero padding by conctinating it with a matrix of zeros
    #     Input: mat - the original one hot matrix, max_seq - the final width of the matrix
    #     Output: concate - the zerro padded matrix
    # =============================================================================
    length = max_seq - mat.shape[0]
    ze = np.zeros((length, mat.shape[1]), dtype='int')
    concate = np.concatenate((mat, ze), axis=0)
    return concate

def oneHot(string, max_seq):
# =============================================================================
#   This function creats one hot encoding to a given DNA string
#     Input: string - the DNA string, max_seq - final length of the sequence
#     Output: one_hot - the encoded matrix
# =============================================================================
        trantab = str.maketrans('ACGT', '0123')
        string = string + 'ACGT'
        data = list(string.translate(trantab))
        mat = to_categorical(data)[0:-4]
        mat = padding(mat, max_seq).flatten()
        return mat

def toFolds(x, y, fold, start):
    # =============================================================================
    #   This function splits a given dataset into test and train sets
    #     Input: x - the complete dataset
    #            y - the complete set of labels
    #            fold - the size of the test set
    #            start - the index in the entire set where the test set starts
    #     Output: x_test, y_test, x_train, y_train
    # =============================================================================
    if start+fold > len(x):
        end = len(x)
    else:
        end = start+fold
    x_test = x[start:end]
    y_test = y[start:end]

    x_train = x.copy()
    [x_train.pop(start) for j in range(start, end)]
    x_train = np.expand_dims(np.asarray(x_train), axis=2)
    x_test = np.expand_dims(np.asarray(x_test), axis=2)
    y_train = np.asarray(y.drop(range(start, end))).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    return x_test, y_test, x_train, y_train
