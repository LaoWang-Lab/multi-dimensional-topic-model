# -*- coding: utf-8 -*-

__author__ = "Linwei Li"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
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
            print len(x), np.shape(n_het), np.shape(axs)
            sns.barplot(x, n_het[h,e,:], palette="Set3", ax=axs[e][h])
            axs[e][h].set_ylabel("counts")
            axs[e][h].set_title("h:%d e:%d" % (h,e))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='input file in json format')
    args = parser.parse_args()

    showTopic(args.json_file)
