import numpy as np
from math import log

def shangnong_gain(dataset):
    num = len(dataset[-1])
    labelcounts = {}
    h = 0.0
    for label in dataset[-1]:
        labelcounts[label] = labelcounts.get(label,0)+1
    for k in labelcounts.values():
        p = k/num
        h -=p*log(p,2)
    return h
