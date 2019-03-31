# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:14:02 2019

@author: LingfengZhao
"""

import numpy as np
from matplotlib import pyplot as plt

gamma = 3 
alpha = np.array([0 ,0 , 1, 1])

x = np.array([0, 1, 2, 3])
y = np.array([1, -1, 1, -1])

z = np.sum(alpha * y * np.exp(- gamma* (x -x[:,None])**2), axis =0)

y_hat = np.where(z > 0, 1, -1)

plt.figure(dpi=200)
plt.plot(x, z, 'x')
plt.plot(x, y_hat, 'o')
plt.legend(["z vs.x", "$\hat{y}$ vs. x"])
plt.xlabel('x')

plt.show()
