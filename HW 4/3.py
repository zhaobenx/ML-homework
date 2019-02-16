import numpy as np
from matplotlib import pyplot as plt
beta_0 = [-1, 2, 1]

x = np.arange(0, 1, 0.1)
y = np.polyval(beta_0, x)

beta_hat = np.polyfit(x, y, 1)
x_plot = np.arange(0, 3, 0.1)
y_pred = np.polyval(beta_hat, x_plot)
y_true = np.polyval(beta_0, x_plot)
plt.plot(x_plot, y_pred)
plt.plot(x_plot, y_true)
plt.plot(x, y, 'o')
plt.legend(['Predicted function', 'True function', 'Train data'])
plt.show()
