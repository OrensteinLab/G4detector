#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:28:56 2020

@author: aliceaubert1
"""
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Activation, Dropout
from keras.layers.pooling import GlobalMaxPooling1D, MaxPool1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.regularizers import l2
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# Initalizing some parameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 1

def oneHot(string):
# =============================================================================
#   This function creates one hot encoding to a given DNA string
#     Input: string - the DNA string
#     Output: mat - the encoded matrix
# =============================================================================
    trantab = str.maketrans('ACGTN', '01234')
    string = string + 'ACGTN'
    data = list(string.translate(trantab))
    mat = to_categorical(data)[0:-5]
    mat = np.delete(mat, -1, -1)
    mat = mat.astype(np.uint8)
    return mat
  
def read_files(pos_f, neg_f):
    #reading from files
    pos = pd.read_csv(pos_f, header=None)[0]
    neg = pd.read_csv(neg_f, header=None)[0]
    
    #filtering out chromosome lines
    chrom = pos[pos.str.contains(">")]
    
    #splitting into chromosome number and window
    chrom = chrom.str.split(':', expand = True)
    
    header = ['chrom', 'window']
    chrom.columns = header
    
    #selecting only chromosome number and reseting index
    chrom = chrom.chrom.reset_index(drop=True)
    
    #filtering out sequence lines and reseting index
    reads = pos[~pos.str.contains(">")]
    reads = reads.reset_index(drop=True)
    
    #making new dataframe of positive examples
    pos = pd.DataFrame({'seq': reads, 'chrom': chrom})
    
    #repeating process for negative examples
    chrom = neg[neg.str.contains(">")]
    chrom = chrom.str.split(':', expand = True)
    header = ['chrom', 'window']
    chrom.columns = header
    chrom = chrom.chrom.reset_index(drop=True)

    reads = neg[~neg.str.contains(">")]
    reads = reads.reset_index(drop=True)
    
    neg = pd.DataFrame({'seq': reads, 'chrom': chrom})
    
    #preparing positive set 
    p = pos[pos.chrom != '>chr1'].seq #filtering out chr1
    p = p.str.upper()
    p = p[~p.str.contains("N")]
    p.reset_index(drop=True, inplace=True)
    lp = np.ones(p.shape[0])
    
    #chromosome 1 test set
    cp = pos[pos.chrom == '>chr1'].seq #selecting chr1
    cp = cp.str.upper()
    cp = cp[~cp.str.contains("N")]
    cp.reset_index(drop=True, inplace=True)
    lcp = np.ones(cp.shape[0])
    
    #negative set
    n = neg[neg.chrom != '>chr1'].seq #filtering out chr1
    n = n.str.upper()
    n = n[~n.str.contains("N")]
    n.reset_index(drop=True, inplace=True)
    ln = np.zeros(n.shape[0])
    
    #chromosome 1 test set (negative)
    cn = neg[neg.chrom == '>chr1'].seq #selecting chr1 
    cn = cn.str.upper()
    cn = cn[~cn.str.contains("N")]
    cn.reset_index(drop=True, inplace=True)
    lcn = np.zeros(cn.shape[0])

    # join and mix
    x = pd.concat([p, n])
    y = np.hstack([lp, ln])
    y = y.astype('int')
    train = pd.DataFrame({'seq': x, 'label': y})
    train = train.sample(frac=1).reset_index(drop=True)

    test_x = pd.concat([cp, cn])
    test_y = np.hstack([lcp, lcn])
    test_y = test_y.astype('int')
    test = pd.DataFrame({'seq': test_x, 'label': test_y})
    test = test.sample(frac=1).reset_index(drop=True)
    
    return train, test

def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                            activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                            bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)

    # Creating Hidden Layer
    hidden1 = Dense(fc)(pool)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=in1, outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    mdl.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae'])
    
    return mdl

def plot(history, window, st, nt):
    # Check out our train accuracy and test accuracy over epochs.
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing accuracy over epochs.
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.plot(test_acc, label='Testing Accuracy', color='red')

    # Set title
    plt.title('{} {} {}nt Fold Training and Testing Accuracy by Epoch'.format(st, nt, str(window)), fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(range(1,16), range(1,16))

    plt.legend(fontsize = 18);
    plt.savefig('plots/{}_{}_{}nt_base_acc'.format(st, nt, str(window)))
    plt.close()
    
    # Check out our train accuracy and test accuracy over epochs.
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing accuracy over epochs.
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(test_loss, label='Testing Loss', color='red')

    # Set title
    plt.title('{} {} {}nt Base Training and Testing Loss by Epoch'.format(st, nt, str(window)), fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Loss', fontsize = 18)
    plt.xticks(range(1,16), range(1,16))

    plt.legend(fontsize = 18);
    plt.savefig('plots/{}_{}_{}nt_base_loss'.format(st, nt, str(window)))
    plt.close()

def aucroc(mdl, x_test, y_test, window, st, nt):
    y_pred_keras = mdl.predict(x_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    
    np.save('plots_arrays/roc/{}_{}_{}nt_base_fpr'.format(st, nt, str(window)), fpr_keras)
    np.save('plots_arrays/roc/{}_{}_{}nt_base_tpr'.format(st, nt, str(window)), tpr_keras)

    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('{} {} {}nt Base ROC curve'.format(st, nt, str(window)))
    plt.legend(loc='best')
    plt.savefig('plots/{}_{}_{}nt_base_auroc'.format(st, nt, str(window)))
    plt.close()

def aupr(mdl, x_test, y_test, window, st, nt):
    y_pred_keras = mdl.predict(x_test).ravel()
    precision, recall, thresholds_keras = precision_recall_curve(y_test, y_pred_keras)
    
    np.save('plots_arrays/pr/{}_{}_{}nt_base_precision'.format(st, nt, str(window)), precision)
    np.save('plots_arrays/pr/{}_{}_{}nt_base_recall'.format(st, nt, str(window)), recall)
    
    score = average_precision_score(y_test, y_pred_keras)

    plt.figure(1)
    plt.plot(recall, precision, label='Keras (area = {:.3f})'.format(score))
    plt.xlabel('Recall rate')
    plt.ylabel('Precision rate')
    plt.title('{} {} {}nt Base Precision Recall curve'.format(st, nt, str(window)))
    plt.legend(loc='best')
    plt.savefig('plots/{}_{}_{}nt_base_aupr'.format(st, nt, str(window)))
    plt.close()    

def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--positive', help='Path to .fa positive examples.',
                        type=str, required=True)
    
    parser.add_argument('-n', '--negative', help='Path to .fa negative examples.',
                        type=str, required=True)
    
    args = parser.parse_args()
    arguments = vars(args)

    return arguments

def main():
    args = user_input()

    if not args['positive']:
        print('use -p to provide a path to an input file for the positive examples')
        sys.exit()
        
    elif not args['negative']:
        print('use -n to provide a path to an input file for the negative examples')
        sys.exit()
   
        
    else:
        print('Reading files\n')
        window = args['positive'].split('.fa')[0].split('_')[-1]
        sample_type = args['positive'].split('.fa')[0].split('_')[2]
        negative_type = args['negative'].split('_')[3]
        
        t1 = time.time()
        train, test = read_files(args['positive'], args['negative'])
        t2 = time.time()
        print(t2-t1)
        
        print('One-Hot Encoding Files\n')
        t1 = time.time()
        x_train=np.array(list(map(oneHot, train.seq)))
        x_test=np.array(list(map(oneHot, test.seq)))
        y_train = np.asarray(train.label)
        y_test = np.asarray(test.label)
        t2 = time.time()
        print(t2-t1)
        
        print('Creating Model\n')
        mdl = model(x_train.shape[1:], window, sample_type, negative_type)
        
        print('Model Training\n')
        t1 = time.time()
        history = mdl.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        t2 = time.time()
        print(t2-t1)
        print('Model Trained!\n')
        
        print('Predicting Labels')
        pred = mdl.predict(x=x_test)

        df = pd.DataFrame(columns=['seq', 'True Label', 'Predicted'])
        df['seq'] = test.seq
        df['True Label'] = test.label
        df['Predicted'] = pred
        
        pred_loc = 'predictions/{}_{}_{}nt_predictions.csv'.format(sample_type, negative_type, str(window))
        
        df.to_csv(pred_loc)
        print('Lables File at {}'.format(pred_loc))

        model_loc = 'models/{}_{}_{}nt_base_model'.format(sample_type, negative_type, str(window))
        mdl.save(model_loc)
        print('Model Saved at: {}'.format(model_loc))
        
        test_scores = mdl.evaluate(x_test, y_test)
        print("Test loss:", test_scores[0])
        print("Test Accuracy:", test_scores[1])
        print("Test MAE:", test_scores[2])
        
        print('Plotting Results\n')
        #plot(history, window, sample_type, negative_type)
        
        aucroc(mdl, x_test, y_test, window, sample_type, negative_type)
        
        aupr(mdl, x_test, y_test, window, sample_type, negative_type)
        

if __name__ == "__main__":
    main()
