# -*- coding: utf-8 -*-

__author__ = "Linwei Li"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os

from settings import outputDir

def plot_fig(json_file, save_fig=False):
    with open(json_file) as f:
        data = json.load(f)
    H, E, M, wot, iter = data['H'], data['E'], data['M'], data['wot'], data['iter']
    fig, axs = plt.subplots(E, H, figsize=(2.5 * E, 3.5 *H), sharex=True)
    fig.suptitle('H:%d E:%d M:%d wot:%d iter:%d' % (H, E, M, wot, iter), fontsize=20, fontweight='bold')
    n_het = np.array(data['topic'])
    y_limits_max = 1.05 * n_het.sum() / (E * H * wot)
    x = np.arange(data['T'])
    for e in range(data['E']):
        for h in range(data['H']):
            # print(len(x), np.shape(n_het), np.shape(axs))
            sns.barplot(x, n_het[h,e,:], palette="Set3", ax=axs[e][h])
            axs[e][h].set_ylabel("counts")
            axs[e][h].set_title("h:%d e:%d" % (h,e))
            axs[e][h].set_ylim([0, y_limits_max])
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_fig:
        i = data['iter']
        plt.savefig('%s/iter%03d.png' % (os.path.dirname(json_file), i), dpi=72, format='png')
    else:
        plt.show()
    plt.close()

def plot_fig_in_dir(json_dir):
     for json_file in os.listdir(json_dir):
        if json_file.endswith('json'):
            plot_fig(json_dir+'/'+json_file, save_fig=True)
            print('ploting %s' % json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--json_file', type=str, help='input file in json format')
    parser.add_argument('-s', '--save_fig', help='save figure and don\'t show', action='store_true')
    parser.add_argument('-d', '--dir', type=str, help='draw all json in dir')
    args = parser.parse_args()

    if args.dir:
        plot_fig_in_dir(args.dir)
    else:
        plot_fig(args.json_file, args.save_fig)
