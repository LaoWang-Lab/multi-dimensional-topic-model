__author__ = 'Linwei'

import numpy as np
import os, json, sys, re

import _cymlda
from settings import H, E, alpha, beta, gamma, docDir, outputDir, iter_max, run_num, dictionary, docset

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
    def __init__(self, H=H, E=E, dictionary=dictionary, docDir=docDir, docset=docset):

        self._dictionary = {x[:-1]:i for (i, x) in enumerate(open(dictionary))}
        self.T = len(self._dictionary)
        self.H = H
        self.E = E

        if docset == 'bagOfWords':
            self._docset = docset
            self._docDir = docDir # when docset is 'bagOfWords', corpus is cotanined in a single file, hence docDir is a filename(not a directory)
            with open(docDir) as f:
                self.M = int(f.readline()) # M value of this corpus is recorded in the first line of corpus file
        else:
            self._docDir = docDir
            self.M = len(os.listdir(docDir))

        self._n_mh = np.zeros((self.M, H), dtype=np.int32) # counts for words in document m which were labeled as in dimension h
        self._n_het = np.zeros((H, E, self.T), dtype=np.int32) # counts for word type t in topic h,e
        self._n_he = np.zeros((H, E), dtype=np.int32) # counts for documents which were labeled as in topic e for dimension h
        self._z_mh = np.random.randint(E, size=(self.M, H)) # value of zi for document m and dimension h
        self._z_mh = np.asarray(self._z_mh, dtype=np.int32)
        self._w_mi = [[],]* self.M # value of wi for document m
        self._s_mi = [[],] * self.M # value of si for document m
        self.n_loaded = 0 # counts for documents loaded

    def readDoc(self, fname):
        m = self.n_loaded
        # self._w_mi[m] = np.genfromtxt(fname, delimiter=',')
        doc = open(fname).read().lower()
        doc = re.sub(r'\n', ' ', doc)
        doc = re.sub(r'-', ' ', doc)
        doc = re.sub(r'[^a-z ]', '', doc)
        doc = re.sub(r' +', ' ', doc)
        words = doc.split()
        words_in_use = filter(lambda x:x in self._dictionary.keys(), words)
        self._w_mi[m] = np.array([self._dictionary[x] for x in words_in_use])
        
        n = len(self._w_mi[m])
        self._n_mh[m] = np.random.multinomial(n, (1/H,)*H)
        self._s_mi[m] = np.random.permutation(n2s(self._n_mh[m]))

        for (s, z) in enumerate(self._z_mh[m]):
            self._n_he[s][z] = self._n_he[s][z] + 1

        for (i, s) in enumerate(self._s_mi[m]):
            self._n_het[s, self._z_mh[m,s], self._w_mi[m][i]] = self._n_het[s, self._z_mh[m,s], self._w_mi[m][i]] + 1

        self.n_loaded += 1

    def list2np(self):
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

    def readCorpus(self):
        if self._docset == 'bagOfWords':

            printFlag = True
            corpusFile = open(self._docDir).readlines()
            self.N_total_words = int(corpusFile[2])
            self._w_mi_ = np.zeros(int(self.N_total_words*5), dtype=np.int32)
            self._n_mw = np.zeros((2, self.M), dtype=np.int32)
            self.cum = 0
            for record in open(self._docDir).readlines()[3:]:
                m, w, counts = [int(x) for x in record.split()]
                m -= 1 # 1 in corpus file is corresponding to 0 in our model
                w -= 1
                self._w_mi_[self.cum:self.cum+counts] = w
                self._n_mw[0,m] += counts
                self.cum += counts
                if m < self.M - 1:
                    self._n_mw[1,m+1] = self.cum
                if m%20 == 0 and printFlag:
                    print('read doc %d' % m)
                    printFlag = False
                elif m%20 != 0:
                    printFlag = True
                # self._w_mi[m].extend([w] * counts)
                # self._w_mi_tmp[m] += [w] * counts
            print('cum is %d' % self.cum)
            self._w_mi_ = self._w_mi_[:self.cum]
            self._s_mi_ = np.random.randint(self.H, size=self.cum)
            self._s_mi_ = np.asarray(self._s_mi_, dtype=np.int32)

            self.n_loaded = m

            for m in range(self.M):
                if m%20 == 0:
                    print('initialize doc %d' % m)
                s_counts = np.bincount(self._s_mi_[self._n_mw[1,m]:self._n_mw[1,m]+self._n_mw[0,m]])
                self._n_mh[m,0:len(s_counts)] = s_counts
                for (s, z) in enumerate(self._z_mh[m]):
                    self._n_he[s][z] += 1

                for i,s in enumerate(self._s_mi_[self._n_mw[1,m]:self._n_mw[1,m]+self._n_mw[0,m]]):
                    self._n_het[s, self._z_mh[m,s], self._w_mi_[i]] += 1

        else:
            for i, docName in enumerate(os.listdir(self._docDir)):
                if i % 20 == 0:
                    print('read docs: %d' % i)
                self.readDoc(os.path.join(self._docDir, docName))
                self.list2np()

    def test_n(self):
        for m in range(self.M):
            for h in range(H):
                assert self._n_mh[m,h] == len(np.where(self._s_mi[m]==h)[0])
        for h in range(H):
            for e in range(E):
                assert self._n_he[h,e] == len(np.where(self._z_mh[:,h]==e)[0])
        test_n_het = np.zeros((H,E,self.T))
        for m in range(self.M):
            for (i,t) in enumerate(self._w_mi[m]):
                h = self._s_mi[m][i]
                test_n_het[h, self._z_mh[m,h], t] += 1
        for h in range(H):
            for e in range(E):
                for t in range(self.T):
                    assert test_n_het[h,e,t] == self._n_het[h,e,t]

    def train_corpus(self, n_iter):
        for i in range(n_iter):
            _cymlda._train_corpus(self._n_het, self._n_he, self._n_mh,
                                  self._w_mi_, self._s_mi_, self._n_mw, self._z_mh,
                                  alpha, beta, gamma)
        # self.test_n()
        # print("sample is OK!")

    def output_topic(self, run_id, iteration):
        _dir = outputDir + os.path.sep + "H%dE%d_M%d" % (H, E, self.M) + os.path.sep + "run%d" % run_id
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        result = {'H':H,'E':E,'M':self.M,'iter':iteration,'T':self.T,'topic':[[[],] * E,]*H,'delta_n_het':self._delta_n_het}
        result['topic'] = self._n_het.tolist()

        with open(os.path.join(_dir, "iter%d.json" % iteration),'w') as f:
            json.dump(result, f)

def run_once(run_id=1):
    print("run_id: %d" % run_id)
    go = mylda()
    go.readCorpus()
    go._n_het_previous = go._n_het.copy()
    go._n_word = go._n_het.sum()
    for i in range(iter_max):
        go.train_corpus(1)
        go._delta_n_het = (np.abs(go._n_het - go._n_het_previous).sum()/go._n_word)
        print("iter %d\t%.10f" % (i, go._delta_n_het))
        go._n_het_previous = go._n_het.copy()
        go.output_topic(run_id, i)

def main():
    run_once(1)

if __name__ == "__main__":
    main()
