__author__ = 'Linwei'

import numpy as np
import os, json, sys
from settings import H, E, M, T, alpha, beta, gamma, wordsOfEachTopic as wot, docDir, outputDir, iter_max

import _cymlda

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange

def n2s(counts):
    """convert a counts vector to corresponding samples"""
    samples = ()
    for (value, count) in enumerate(counts):
        samples = samples + (value,)*count
    return samples

class mylda:
    def __init__(self, H=H, E=E, M=M, T=T):
        self._n_mh = np.zeros((M, H)) # counts for words in document m which were labeled as in dimension h
        self._n_het = np.zeros((H, E, T)) # counts for word type t in topic h,e
        self._n_he = np.zeros((H, E)) # counts for documents which were labeled as in topic e for dimension h
        self._z_mh = np.random.randint(E, size=(M, H)) # value of zi for document m and dimension h
        self._w_mi = [[],]* M # value of wi for document m
        self._s_mi = [[],] * M # value of si for document m
        self.n_loaded = 0 # counts for documents loaded

    def readDoc(self, fname):
        m = self.n_loaded
        self._w_mi[m] = np.genfromtxt(fname, delimiter=',')
        n = len(self._w_mi[m])
        self._n_mh[m] = np.random.multinomial(n, (1/H,)*H)
        self._z_mh[m] = np.random.randint(E, size=H)
        self._s_mi[m] = np.random.permutation(n2s(self._n_mh[m]))

        for (s, z) in enumerate(self._z_mh[m]):
            self._n_he[s][z] = self._n_he[s][z] + 1

        for (i, s) in enumerate(self._s_mi[m]):
            self._n_het[s, self._z_mh[m,s], self._w_mi[m][i]] = self._n_het[s, self._z_mh[m,s], self._w_mi[m][i]] + 1

        self.n_loaded = self.n_loaded + 1

    def readCorpus(self):
        for i in range(M):
            self.readDoc(docDir + os.path.sep + "%d.dat"%i)

        self._n_mh = self._n_mh.astype(dtype=np.int32)
        self._n_he = self._n_he.astype(dtype=np.int32)
        self._n_het = self._n_het.astype(dtype=np.int32)
        self._z_mh = self._z_mh.astype(dtype=np.int32)

        nmw = [None, None]
        nmw[0] = [len(doc) for doc in self._w_mi]
        nmw[1] = [0] + nmw[0][:-1]
        cum = 0
        for i in range(len(nmw[1])):
            cum += nmw[1][i]
            nmw[1][i] = cum
        self._n_mw = np.asarray(nmw, dtype=np.int32)

        self._w_mi_ = []
        for i in self._w_mi:
            self._w_mi_.extend(i)
        self._w_mi_ = np.asarray(self._w_mi_, dtype=np.int32)

        self._s_mi_ = []
        for i in self._s_mi:
            self._s_mi_.extend(i)
        self._s_mi_ = np.asarray(self._s_mi_, dtype=np.int32)

    def test_n(self):
        for m in range(M):
            for h in range(H):
                assert self._n_mh[m,h] == len(np.where(self._s_mi[m]==h)[0])
        for h in range(H):
            for e in range(E):
                assert self._n_he[h,e] == len(np.where(self._z_mh[:,h]==e)[0])
        test_n_het = np.zeros((H,E,T))
        for m in range(M):
            for (i,t) in enumerate(self._w_mi[m]):
                h = self._s_mi[m][i]
                test_n_het[h, self._z_mh[m,h], t] += 1
        for h in range(H):
            for e in range(E):
                for t in range(T):
                    assert test_n_het[h,e,t] == self._n_het[h,e,t]

    def train_corpus(self, n_iter):
        for i in range(n_iter):
            _cymlda._train_corpus(self._n_het, self._n_he, self._n_mh,
                                  self._w_mi_, self._s_mi_, self._n_mw, self._z_mh,
                                  alpha, beta, gamma)
        # self.test_n()
        # print("sample is OK!")

    def output_topic(self, iteration):
        _dir = outputDir + os.path.sep + "H%dE%d_wot%d_M%d" % (H, E, wot, M) + os.path.sep
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        result = {'H':H,'E':E,'M':M,'wot':wot,'iter':iteration,'T':T,'topic':[[[],] * E,]*H,'delta_n_het':self._delta_n_het}
        result['topic'] = self._n_het.tolist()

        with open(_dir + "iter%d.json" % iteration,'w') as f:
            json.dump(result, f)


def main():
    go = mylda()
    go.readCorpus()
    go._n_het_previous = go._n_het.copy()
    go._n_word = go._n_het.sum()
    countdown = 10
    for i in range(iter_max):
        go.train_corpus(1)
        go._delta_n_het = (np.abs(go._n_het - go._n_het_previous).sum()/go._n_word)
        print("iter %d\t" % i, go._delta_n_het)
        go._n_het_previous = go._n_het.copy()
        if i%1 == 0:
            go.output_topic(i)

        if go._delta_n_het < 1e-4:
            countdown -= 1
        if countdown == 0:
            break    
            
if __name__ == "__main__":
    main()
