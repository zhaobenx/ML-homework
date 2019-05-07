# %%
import numpy as np

# %%
X = np.array([[3, 2, 1], [2, 4, 5], [1, 2, 3], [0, 2, 5]])
mean = X.mean(axis=0)
print(mean)

# %%
Q = np.cov(X.T, bias=True)
print(X - mean)
print(Q)

# %%
eigval, eigvec = np.linalg.eigh(Q)
print(eigval)
print(eigvec)

# %%
# print(np.dot(np.linalg.inv(vec), vec))
# print(np.linalg.det(vec))
# print(np.dot(vec[:,0],vec[:,1]))
a = np.dot(X - mean, eigvec)
print(a)

# %%
rec = np.dot(a, eigvec.T) + mean
print(rec)

# %%
two_large_eigvec = eigvec[:, 1:]  # last two eignvalues are the biggest
a2 = np.dot(X - mean, two_large_eigvec)
rec2 = np.dot(a2, two_large_eigvec.T) + mean
print(rec2)

# %%
error = np.sum((rec2 - X)**2)
print(error)
