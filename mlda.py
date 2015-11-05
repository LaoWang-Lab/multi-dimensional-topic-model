__author__ = 'Linwei'

import numpy as np
import os, json
from settings import H, E, M, T, alpha, beta, gamma, wordsOfEachTopic as wot, docDir, outputDir

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

    def sampleDim(self, m, i):
        hi = self._s_mi[m][i]
        ti = self._w_mi[m][i]
        zi = self._z_mh[m][hi]
        self._n_mh[m, hi] -= 1
        if self._n_het[hi, self._z_mh[m, hi], ti] == 0:
            raise Exception("sssss")
        self._n_het[hi, zi, ti] -= 1
        # to be continue
        theta = (self._n_mh[m] + alpha) / (self._n_mh[m] + alpha).sum()
        get_phi_ht = lambda h:(self._n_het[h, self._z_mh[m, h], ti]+beta)/(self._n_het[h, self._z_mh[m, h], :]+beta).sum()
        phi_t = np.array([get_phi_ht(h) for h in range(H)])
        tmp = (theta*phi_t)/(theta*phi_t).sum()
        hi = np.random.multinomial(1, tmp)
        hi = np.nonzero(hi)[0][0]
        zi = self._z_mh[m, hi]
        self._n_mh[m, hi] += 1
        self._n_het[hi, zi, ti] += 1
        self._s_mi[m][i] = hi

    def get_wI(self, m, h):
        for (i, s) in enumerate(self._s_mi[m]):
            if s == h:
                yield i

    def sampleTopic(self, m, h):
        si = h
        zi = self._z_mh[m, si]
        self._n_he[si, zi] = self._n_he[si, zi] - 1
        wI = list(self.get_wI(m, si))
        for iw in wI:
            self._n_het[si, zi, self._w_mi[m][iw]] -= 1

        def pZ(e):
            #result = (self._n_he[si, e] + gamma)/(self._n_he[si, :] + gamma).sum()
            result = (self._n_he[si, e] + gamma)/(self._n_he[si, :] + gamma).sum()
            for iw in wI:
                ti = self._w_mi[m][iw]
                pTi = (self._n_het[si, e, ti] + beta)/(self._n_het[si, e, :] + beta).sum()
                result = result * pTi
            return result

        tmp = np.array([float(pZ(e)) for e in range(E)])
        zi = np.random.multinomial(1, tmp/tmp.sum())
        #zi = np.random.multinomial(1, [pZ(e) for e in range(E)])
        zi = np.nonzero(zi)[0][0]
        self._z_mh[m, si] = zi
        self._n_he[si, zi] += 1
        for iw in wI:
            self._n_het[si, zi, self._w_mi[m][iw]] += 1

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

    def train_doc(self, m):
        for i,wi in enumerate(self._w_mi[m]):
            self.sampleDim(m, i)
        for h in range(H):
            self.sampleTopic(m, h)

    def train_corpus(self, n_iter):
        for i in range(n_iter):
            for m in range(M):
                self.train_doc(m)
        # self.test_n()
        # print("sample is OK!")

    def output_topic(self, iteration):
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        result = {'H':H,'E':E,'M':M,'wot':wot,'iter':iteration,'topic':[[[],] * E,]*H}
        for h in range(H):
            for e in range(E):
                result['topic'][h][e] = list(self._n_het[h,e,:])

        with open(outputDir + os.path.sep + "H%dE%d_wot%d_M%d_iter%d.json" % (H, E, wot, M, iteration),'w') as f:
            json.dump(result, f)


if __name__ == "__main__":
    go = mylda()
    go.readCorpus()
    for i in range(1000):
        go.train_corpus(1)
        if i%5 == 0:
            go.output_topic(i)

