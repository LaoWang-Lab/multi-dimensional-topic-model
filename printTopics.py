# -*- coding: utf-8 -*-
# Last modified: Wed Nov 18 20:32:02 2015

__author__ = "Linwei Li"

import json
import numpy as np
import argparse

from settings import dictionary

def printTopic(jsonFile, dictionary=dictionary, topNum=15):
    dictionary = [x[:-1] for x in open(dictionary)]
    with open(jsonFile) as f:
        data = json.load(f)
    H, E, topic = data['H'], data['E'], np.array(data['topic'])
    with open('H%d_E%d_M%d_topic.txt' % (H, E, data['M']), 'wt') as f:
        for h in range(H):
            f.write('h : %d\n' % h)
            for e in range(E):
                f.write('e%d : ' % e)
                topWords = topic[h, e, :].argsort()[::-1][:topNum]
                for t in topWords:
                    f.write('%s:%d  ' % (dictionary[t], topic[h,e,t]))
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--jsonFile', help='the file stores the result of topic', required=True)
    parser.add_argument('-n', '--topNum', type=int, help='the number of words printed for each topic', default=15)    
    args = parser.parse_args()
    printTopic(jsonFile=args.jsonFile, topNum=args.topNum)
     
