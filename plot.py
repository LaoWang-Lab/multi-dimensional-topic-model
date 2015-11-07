# -*- coding: utf-8 -*-

__author__ = "Linwei Li"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

from settings import outputDir

def showTopic(jsonFile):
    with open(jsonFile) as f:
        data = json.load(f)
    H, E, M, wot, iter = data['H'], data['E'], data['M'], data['wot'], data['iter']
    fig, axs = plt.subplots(E, H, figsize=(2.5 * E, 3.5 *H), sharex=True)
    fig.suptitle('H:%d E:%d M:%d wot:%d iter:%d' % (H, E, M, wot, iter), fontsize=20, fontweight='bold')
    n_het = np.array(data['topic'])
    x = np.arange(data['T'])
    for e in range(data['E']):
        for h in range(data['H']):
            # print(len(x), np.shape(n_het), np.shape(axs))
            sns.barplot(x, n_het[h,e,:], palette="Set3", ax=axs[e][h])
            axs[e][h].set_ylabel("counts")
            axs[e][h].set_title("h:%d e:%d" % (h,e))
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='input file in json format')
    args = parser.parse_args()

    showTopic(args.json_file)
