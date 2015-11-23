# -*- coding: utf-8 -*-
# Last modified: Fri Oct 30 00:29:47 2015

__author__ = "Linwei Li"

H = 2
E = 10

alpha = 1
beta = 0.1
gamma = 10

docDir = 'dataset/docword.enron.txt'
outputDir = 'trainedTopic'
dictionary = 'dataset/vocab.enron.txt'
docset = 'bagOfWords'
iter_max = 500
run_num = 10

try:
    from localSettings import *
except:
    pass
