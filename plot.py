# -*- coding: utf-8 -*-

__author__ = "Linwei Li"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from settings import outputDir

def showTopic(jsonFile):
    with open(jsonFile) as f:
        data = json.load(f)
    fig, axs = plt.subplots(data['E'], data['H'], figsize=(8, 6), sharex=True)
    for e in range(data['E']):
        # to be continued
