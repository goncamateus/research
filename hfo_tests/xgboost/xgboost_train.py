import pickle

import numpy as np

import xgboost as xgb

param = {'max_depth': 20, 'eta': 1,
         'silent': 1, 'objective': 'binary:logistic'}
num_round = 60

for unum in range(2, 9):
    print('Agent', unum)
    states = pickle.load(open('states_{}.dump'.format(unum), 'rb'))
    actions = pickle.load(open('actions_{}.dump'.format(unum), 'rb'))
    dtrain = xgb.DMatrix(states[:-1778], label=actions[:-1778])
    dtest = xgb.DMatrix(states[-1778:], label=actions[-1778:])
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    bst.save_model('agent_{}.model'.format(unum))
