import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

UNUM = 2


def train(X_train, Y_train):
    global UNUM
    svc = SVC(kernel='rbf', gamma="scale")
    svc.fit(X_train, Y_train)

    fiile = open('svm_{}.dump'.format(UNUM), 'wb+')
    pickle.dump(svc, fiile)

    return svc


def test(model, X_test, Y_test):
    return model.score(X_test, Y_test)


def main():
    Y = []
    x = []

    tr = True
    global UNUM
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s : %(message)s',
                        handlers=[logging.FileHandler('svm.log'),
                                  logging.StreamHandler()])

    for i in [2, 3, 4]:
        UNUM = i
        df = pd.read_csv('svm_db_{}.csv'.format(UNUM))
        states = np.array(df.iloc[:, :-1])
        actions = df.iloc[:, -1]

        print('Databases loaded')

        x_train = states[:-2000]
        y_train = actions[:-2000]
        x_test = states[-2000:]
        y_test = actions[-2000:]

        if tr:
            model = train(x_train, y_train)
        else:
            fiile = open('svm_{}.dump'.format(UNUM), 'rb')
            model = pickle.load(fiile)

        score = test(model, x_test, y_test)
        logging.info('%d: %d', UNUM, score)


if __name__ == "__main__":
    main()
