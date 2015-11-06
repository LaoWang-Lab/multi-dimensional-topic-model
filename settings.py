# -*- coding: utf-8 -*-
# Last modified: Fri Oct 30 00:29:47 2015

__author__ = "Linwei Li"

H = 2
E = 3
M = 600
N = 200

alpha = 1
beta = 0.1
gamma = 10

wordsOfEachTopic = 2

docDir = 'docSrc'
outputDir = 'trainedTopic'

try:
    from localSettings import *
except:
    pass

T = H * E * wordsOfEachTopic
