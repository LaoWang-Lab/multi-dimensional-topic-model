#  -*- coding: utf-8 -*-
# Last modified: Fri Oct 30 00:29:11 2015

__author__ = "Linwei Li"

import numpy as np
import random, os
from settings import H, E, N, M, T, wordsOfEachTopic as wot, alpha, beta, gamma, docDir, pureness

def get_k(s, z):
    return s * E + z

def n2s(counts):
    """convert a counts vector to corresponding samples"""
    samples = ()
    for (value, count) in enumerate(counts):
        samples = samples + (value,)*count
    return samples

def get_t(k=None, s=None, z=None):
    if k is not None:
        (s, z) = int(k/E), k % E
    return (s*E+z) * wot + np.arange(wot)

def get_topic(H=H, E=E, wot=wot, T=T, pureness=pureness):
    phi = np.ones((H,E,T)) * (1-pureness) / (T-wot)
    for h in range(H):
        for e in range(E):
            for i in range(wot):
                phi[h,e,(h*E+e)*wot+i] = pureness/wot
    return phi

def gen_turbid_document(m, prefix=docDir, pureness=pureness):
    choosedTopics = np.random.randint(E, size=H)
    phi = get_topic()
    theta = np.random.dirichlet((alpha,)*H)
    n = np.random.poisson(N)
    dimensionCounts = np.random.multinomial(n, theta)
    word_counts = np.zeros(T)
    for h in range(H):
        # add = np.random.multinomial(dimensionCounts[h], phi[h, choosedTopics[h], :]k)
        word_counts += np.random.multinomial(dimensionCounts[h], phi[h, choosedTopics[h], :])
    docGen = np.random.permutation(n2s(word_counts))

    with open(prefix + os.path.sep + "%d.dat" % m, 'wt') as f:
        f.write(str(list(docGen))[1:-1])

def gen_corpus(docDir=docDir):
    if not os.path.exists(docDir):
        os.mkdir(docDir)
    for m in range(M):
        # gen_document(m)
        gen_turbid_document(m)


if __name__ == "__main__":
    gen_corpus(docDir=docDir)
