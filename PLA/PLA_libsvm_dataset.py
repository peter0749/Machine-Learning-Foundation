
# coding: utf-8

# In[1]:

import sys
import numpy as np


# In[2]:


class PLA(object):
    def __init__(self, x_dim, eta=1.0, shuffle=False, verbose=False):
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta = eta
        self.Wxb = np.random.normal(0, np.sqrt(2/(x_dim+1)), size=(1,x_dim+1)) # initialize Wxb using he-normal
    def predict(self, x, pocket=False):
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(self.Wxb @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        correct_cnt = 0
        i = 0
        while correct_cnt<len(Xs): # cyclic method
            if self.shuffle and correct_cnt==0:
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
                i = 0
            x, y = Xs[i], Ys[i]
            p = self.predict(x)
            if p!=y: # wrong
                self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                updates += 1
                if self.verbose:
                    print('iteration {:d}: '.format(updates), self.Wxb)
                correct_cnt = 0
            else:
                correct_cnt += 1
            i = (i+1)%len(Xs)
        return updates


# In[3]:


class PocketPLA(PLA):
    def __init__(self, x_dim, eta=1.0, pocket_maxiter=None, shuffle=False, verbose=False):
        super(PocketPLA, self).__init__(x_dim, eta=eta, shuffle=shuffle, verbose=verbose)
        self.pocket_maxiter = pocket_maxiter
        self.Wxb_pocket = np.zeros_like(self.Wxb, dtype=np.float32) # (1, 4)
    def predict(self, x, pocket=False):
        W = self.Wxb_pocket if pocket else self.Wxb
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(W @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        last_errors = np.inf
        while True:
            if self.shuffle: # precomputed random order; else: naive cyclic
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                if p!=y: # wrong
                    self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                    updates += 1
                    break
            errors = 0
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                errors += 1 if p!=y else 0
            if errors < last_errors:
                last_errors = errors
                self.Wxb_pocket = self.Wxb.copy()
                if self.verbose:
                    print('iteration {:d}: update pocket weights: err: {:.2f}'.format(updates, errors/len(Xs)))
            if updates>=self.pocket_maxiter:
                return last_errors


# In[4]:


def data_reader(filepath):
    with open(filepath, 'r') as fp:
        x = []
        y = []
        for line in fp:
            split_line = line.split()
            y.append(split_line[0])
            feature = np.asarray([f.split(':')[1] for f in split_line[1:]], dtype=np.float32)
            x.append(feature)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    labels = sorted(list(set(y)))
    assert len(labels)==2
    y[y==labels[0]] =-1
    y[y==labels[1]] = 1
    y = y.astype(np.int16)
    return x, y

# In[5]:

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
max_ite    = int(sys.argv[3])

# In[6]:

x_train, y_train = data_reader(train_path)
x_test, y_test = data_reader(test_path)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

pla = PocketPLA(x_train.shape[-1], pocket_maxiter=max_ite, shuffle=True, verbose=True)
pla.train(x_train, y_train)
preds = np.squeeze(np.asarray([pla.predict(x, pocket=True) for x in x_test]))
err = (preds!=y_test).mean()
print('error rate: {:.2f}'.format(err))
print('accuracy: {:.2f}'.format(1-err))

