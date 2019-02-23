from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.random.rand(30, 5)
y = X[:, [4]] * 2 + X[:, [2]] * 6
Xtr, Xts, ytr,  yts = train_test_split(X, y)


scores = []
columns = []
for i in combinations(range(Xtr.shape[1]), 2):
    model = LinearRegression()  # Create a linear regression model object
    model.fit(Xtr[:, i], ytr)  # Fits the model
    yhat = model.predict(Xts[:, i])
    scores.append(np.mean((yhat - yts)**2))
    columns.append(i)

print(f"Best model is order {columns[np.argmin(scores)]}, rss is {np.min(scores)}")
