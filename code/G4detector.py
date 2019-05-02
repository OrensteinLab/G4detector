import sys
import pandas as pd
import numpy as np
import utils_GGG as ug
from keras import backend as K
from model import model as mdl
from sklearn.metrics import roc_curve, auc
from keras.models import load_model

num_epoch = 15
filter_size = (80, 80, 96)
opt_func = 'Adam'
lr = 1e-4
batch_size = 128
hidden = 32
n_folds = 10
meanAUC = []
stdAUC = []


#perform cross validatio
if sys.argv[0] == "cross-val":

    # prepare positive set
    p = pd.read_csv(sys.argv[1], header=None)
    p = p[0]
    p = p[~p.str.contains("N")]
    p = p.str.upper()
    p.reset_index(drop=True, inplace=True)
    lp = np.ones(p.shape[0])

    # prepare nagative set
    n = pd.read_csv(sys.argv[2], header=None)
    n = n[0]
    n = n[~n.str.contains("N")]
    n.reset_index(drop=True, inplace=True)
    ln = np.zeros(n.shape[0])

    # join and mix
    x = pd.concat([p, n])
    y = np.hstack([lp, ln])
    y = y.astype('int')
    joint = pd.DataFrame({'seq': x, 'label': y})
    joint = joint.sample(frac=1).reset_index(drop=True)
    l = x.apply(len)
    max_seq = np.max(l)

    x_oh = [ug.oneHot(string, max_seq=max_seq) for string in joint['seq']]  # turn to one-hot encoded representation

    fold = int(np.ceil(joint.shape[0] / n_folds))
    AUC = []

    for start in np.arange(0, joint.shape[0], fold):
        x_test, y_test, x_train, y_train = ug.toFolds(x_oh, joint['label'], fold, start)
        input_shape = (x_train.shape[1], 1)
        model = mdl(input_shape, filter_size=filter_size, opt_func=opt_func, lr=lr, fc=hidden)
        history = model.fit(x=x_train, y=y_train,
                            validation_data=(x_test, y_test),
                            batch_size=batch_size, epochs=num_epoch, verbose=1)

        score = model.predict(x_test, batch_size=batch_size)

        fpr, tpr, thresh = roc_curve(y_test, score, pos_label=1)
        AUC.append(auc(fpr, tpr))

        K.clear_session()

    print(np.mean(AUC))
    print(np.std(AUC))

elif sys.argv[0] == 'train': #train on complete dataset

    # prepare positive set
    p = pd.read_csv(sys.argv[1], header=None)
    p = p[0]
    p = p[~p.str.contains("N")]
    p = p.str.upper()
    p.reset_index(drop=True, inplace=True)
    lp = np.ones(p.shape[0])

    # prepare nagative set
    n = pd.read_csv(sys.argv[2], header=None)
    n = n[0]
    n = n[~n.str.contains("N")]
    n.reset_index(drop=True, inplace=True)
    ln = np.zeros(n.shape[0])

    # join and mix
    x = pd.concat([p, n])
    y = np.hstack([lp, ln])
    y = y.astype('int')
    joint = pd.DataFrame({'seq': x, 'label': y})
    joint = joint.sample(frac=1).reset_index(drop=True)
    l = x.apply(len)
    max_seq = np.max(l)

    x_oh = [ug.oneHot(string, max_seq=max_seq) for string in joint['seq']]  # turn to one-hot encoded representation

    x_train = np.expand_dims(np.asarray(x_oh), axis=2)
    y_train = np.asarray(joint['label']).reshape(-1, 1)
    input_shape = (x_train.shape[1], 1)

    model = mdl(input_shape, filter_size=filter_size, opt_func=opt_func, lr=lr, fc=hidden)
    history = model.fit(x=x_train, y=y_train,
                        batch_size=batch_size, epochs=num_epoch, verbose=1)

    model.save('G4detector_model.h5')
    K.clear_session()

else: #test with existing model

    x = pd.read_csv(sys.argv[1], header=None) #load sequences
    x = x[0]
    x = x[~x.str.contains("N")]
    x = x.str.upper()
    x.reset_index(drop=True, inplace=True)

    l = x.apply(len)
    model = load_model(sys.argv[2])
    max_seq = int(model._feed_input_shapes[0][1]/4) #remove sequences that are longer the the max sequence the model was trained on
    x_Test = x[l <= max_seq]
    x_oh_test = [ug.oneHot(string, max_seq=max_seq) for string in x_Test]
    x_test = np.expand_dims(np.asarray(x_oh_test), axis=2)

    score = model.predict(x_test, batch_size=batch_size)
    score_list = [a for b in score.tolist() for a in b]
    df3 = pd.DataFrame({"seq": x_Test, "score": score_list})
    df3.to_csv("G4detector_scores.csv")