# -*- coding: utf-8 -*-
# Last modified: Fri Oct 30 00:29:11 2015

__author__ = "Linwei Li"

import numpy as np
import random
from settings import H, E, N, M, T, wordsOfEachTopic as wot

def get_k(s, z):
    return s * E + z

def get_t(k=None, s=None, z=None):
    if k is not None:
        (s, z) = int(k/E), k % E
    return (s*E+z) * wot + np.arange(wot)

def gen_document(m):
    choosedTopics = np.random.randint(E, size=H)
    theta = np.random.dirichlet((0.2,)*H)
    n = np.random.poisson(N)
    dimensionCounts = np.random.multinomial(n, theta)
    samples = ()
    for (s, z) in enumerate(choosedTopics):
        for t in get_t(s=s, z=z):
            samples = samples + (t,) * dimensionCounts[s]
    docGen = random.sample(samples, len(samples))

    with open("data/%d.dat" % m, 'wt') as f:
        f.write(str(docGen)[1:-1])

def main():
    import os
    import shutil
    if 'data' not in os.listdir('.'):
        os.mkdir('data')
    else:
        shutil.rmtree('data')
        os.mkdir('data')
    for m in range(M):
        gen_document(m)

if __name__ == "__main__":
    main()
