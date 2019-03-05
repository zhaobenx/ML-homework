import numpy as np
from matplotlib import pyplot as plt

incomes = np.array([30, 50, 70, 80, 100])
num_websites = np.array([0, 1, 1, 2, 1])
donate = np.array([0, 1, 0, 1, 1], dtype=bool)


plt.scatter(incomes[donate], num_websites[donate], c='red')
plt.scatter(incomes[~donate], num_websites[~donate], c='blue')
plt.legend(['Donate', 'Not donate'])
plt.xlabel('Income (thousands $)')
plt.ylabel('Num websites')
plt.show()
