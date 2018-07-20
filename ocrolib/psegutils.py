#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function

from toplevel import *
from pylab import *
from scipy.ndimage import filters,interpolation
import sl,morph

def B(a):
    if a.dtype==dtype('B'): return a
    return array(a,'B')

class record:
    def __init__(self,**kw): self.__dict__.update(kw)
        
def binary_objects(binary):
    labels,n = morph.label(binary)
    objects = morph.find_objects(labels)
    return objects

def estimate_scale(binary):
    '''估计文本行宽度'''
    objects = binary_objects(binary)

    bysize = sorted(objects, key=sl.area)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o]) > 0:
            continue
        scalemap[o] = sl.area(o) ** 0.5
    scale = median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale

