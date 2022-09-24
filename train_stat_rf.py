import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import pickle
import pdb

from data_stat_cpu import data_train, data_test, num_input_features, num_outputs

from sklearn.metrics import r2_score


def test(model):
    print('======= Training dataset ======')
    for k, (x, y) in data_train.items():
        print(k, r2_score(model.predict(x), y))
    print('======= Test dataset ======')
    for k, (x, y) in data_test.items():
        print(k, r2_score(model.predict(x), y))

def train(model):
    model = RandomForestRegressor(verbose=1, n_jobs=48)
    x, y = data_train_ensemble
    model.fit(x, y)
    with open('./checkpoints/netstat_rf.pickle', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    with open('./checkpoints/netstat_rf.pickle', 'rb') as f:
        model = pickle.load(f)
    model.verbose = 0
    test(model)
    # train(model)
