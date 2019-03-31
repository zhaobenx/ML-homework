# -*- coding: utf-8 -*-
"""
Created on 2019-03-31 17:26:26
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import numpy as np
from matplotlib import pyplot as plt

x = np.fromstring("0 1.3 2.1 2.8 4.2 5.7", sep=' ')
y = np.fromstring("-1 -1 -1 1 -1 1", sep=' ')


t = np.linspace(0, 5, 100)[:, None]
z = x - t
y_hat = np.where(z > 0, 1, -1)

J = np.sum(np.maximum(0, 1 - y*z), axis=1)

plt.figure(dpi=200)
plt.plot(t, J)
plt.xlabel('t')
plt.ylabel('J(t)')
plt.show()
