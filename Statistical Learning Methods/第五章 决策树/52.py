# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:42:43 2017

@author: WZW
"""

import numpy as np

def cutPoint(arr, start, size):
    err = [0] * (size-1)
    for i in range(start+1, start+size):
        arr1 = arr[start:i]
        arr2 = arr[i:start+size]
        err[i-1-start] = (arr1.std()**2)*arr1.size + (arr2.std()**2)*arr2.size

    print (err)
    print (min(err))
    return err.index(min(err))+start

arr = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9])
start = 0
size = 10
index = cutPoint(arr, start, size)

print (index, arr[index], arr[start:(index+1)].mean(), arr[(index+1):start+size].mean())