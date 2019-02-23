from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.random.rand(30, 5)
y = X[:, [3]] * 2
Xtr, Xts, ytr,  yts = train_test_split(X, y)


scores = []
for i in range(Xtr.shape[1]):
    model = LinearRegression()  # Create a linear regression model object
    model.fit(Xtr[:, [i]], ytr)  # Fits the model
    yhat = model.predict(Xts[:, [i]])
    scores.append(np.mean((yhat - yts)**2))

print(f"Best model is order {np.argmin(scores)}, rss is {np.min(scores)}")
