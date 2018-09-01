# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

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
                    x2_start = (-self.Wxb[0,0]*x1_min-self.Wxb[0,-1]) / self.Wxb[0,1]
                    x2_end   = (-self.Wxb[0,0]*x1_max-self.Wxb[0,-1]) / self.Wxb[0,1]
                    if 'line' in locals(): line.pop(0).remove()
                    line = plt.plot([x1_min, x1_max], [x2_start, x2_end], color='black')
                    plt.pause(0.05)
                correct_cnt = 0
            else:
                correct_cnt += 1
            i = (i+1)%len(Xs)
        return updates

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
                    x2_start = (-self.Wxb[0,0]*x1_min-self.Wxb[0,-1]) / self.Wxb[0,1]
                    x2_end   = (-self.Wxb[0,0]*x1_max-self.Wxb[0,-1]) / self.Wxb[0,1]
                    if 'line1' in locals(): line1.pop(0).remove()
                    line1 = plt.plot([x1_min, x1_max], [x2_start, x2_end], color='black')
                    plt.pause(0.05)
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
                    x2_start = (-self.Wxb[0,0]*x1_min-self.Wxb[0,-1]) / self.Wxb[0,1]
                    x2_end   = (-self.Wxb[0,0]*x1_max-self.Wxb[0,-1]) / self.Wxb[0,1]
                    if 'line2' in locals(): line2.pop(0).remove()
                    line2 = plt.plot([x1_min, x1_max], [x2_start, x2_end], color='green')
                    plt.pause(0.05)
            if updates>=self.pocket_maxiter or last_errors==0:
                return last_errors

N = 1000
x1 = np.random.normal((-5, 10), (5,5),  size=(N,2))
x2 = np.random.normal((10, -5), (5,3),  size=(N,2))
x  = np.append(x1, x2, axis=0)
y  = np.append(np.ones(N), -np.ones(N))
idx = np.random.permutation(len(x))
x, y = x[idx], y[idx]
x1_min, x1_max = x1.min(), x1.max()
x2_min, x2_max = x2.min(), x2.max()

plt.scatter(x1[...,0], x1[...,1], color='blue', marker='o')
plt.scatter(x2[...,0], x2[...,1], color='red', marker='x')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.ion()
pla = PocketPLA(x.shape[-1], pocket_maxiter=100, shuffle=True, verbose=True)
pla.train(x, y)
plt.ioff(); plt.show()

