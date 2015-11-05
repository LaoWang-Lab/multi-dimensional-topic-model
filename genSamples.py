#  -*- coding: utf-8 -*-
# Last modified: Fri Oct 30 00:29:11 2015

__author__ = "Linwei Li"

import numpy as np
import random, os
from settings import H, E, N, M, T, wordsOfEachTopic as wot, alpha, beta, gamma, docDir

def get_k(s, z):
    return s * E + z

def get_t(k=None, s=None, z=None):
    if k is not None:
        (s, z) = int(k/E), k % E
    return (s*E+z) * wot + np.arange(wot)

def gen_document(m, prefix=docDir):
    choosedTopics = np.random.randint(E, size=H)
    theta = np.random.dirichlet((alpha,)*H)
    n = np.random.poisson(N)
    dimensionCounts = np.random.multinomial(n, theta)
    samples = ()
    for (s, z) in enumerate(choosedTopics):
        for t in get_t(s=s, z=z):
            samples = samples + (t,) * dimensionCounts[s]
    docGen = random.sample(samples, len(samples))

    with open(prefix + os.path.sep + "%d.dat" % m, 'wt') as f:
        f.write(str(docGen)[1:-1])
                # to be continued
                # docGen = random.sample()
                # f.write("%d %d\n"%(t, topics_counts[s]))

if __name__ == "__main__":
    if not os.path.exists(docDir):
        os.mkdir(docDir)
    for m in range(M):
        gen_document(m)

