# -*- coding: utf-8 -*-

__author__ = "Linwei Li"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from settings import H, E, M, wordsOfEachTopic as wot, T

from settings import outputDir

def showTopic(jsonFile):
    with open(jsonFile) as f:
        data = json.load(f)
    fig, axs = plt.subplots(data['E'], data['H'], figsize=(8, 6), sharex=True)
    fig.suptitle('H:%d E:%d M:%d wot:%d' % (H,E,M,wot))
    n_het = np.array(data['topic'])
    x = np.arange(T)
    for e in range(data['E']):
        for h in range(data['H']):
            sns.barplot(x, n_het[h,e,:], palette="Set3", ax=axs[e][h])
            axs[e][h].set_ylabel("counts")
            axs[e][h].set_title("h:%d e:%d" % (h,e))
    plt.show()
