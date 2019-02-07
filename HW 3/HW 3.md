# Homework3

Lingfeng Zhao

## 1. 

### (a)

Target variable is sales trends( history ).

### (b)

Give some scores to the judgement words. E.g. give each comment initial point 0, more than three "good" add 10 points, less than three "good" add 5 points; more than three "bad" and "doesn't work" minus 10 points, less than three minus 5 points.

Use numeric score and this judgments score as two attributes of the multiple linear regression.

### (c)

The score can be normalized, so it ranges from 0.0 to 1.0.

### (d)

The features can be adjusted as follows:

Good makes score of 5, bad makes score of 1. No rating makes the score 2.5.

### (e)

I would choose to use the fraction, to make the comparison more obivouse.



## 3.

### (a)

$$
\hat{y} = ( a_1x_1 +  a_2x_2)e^{-x_1-x_2}
$$
$$
\pmb\beta = [a_1, a_2 ]
$$
$$
\pmb\phi(x_1, x_2) = [ x_1e^{-x_1-x_2}, x_2e^{-x_1-x_2} ]
$$

### (b)

$$
\hat{y} = \begin{cases}
    a_1 + a_2x & \mbox{if } x < 1 \\
    a_3 + a_4x & \mbox{if } x \geq 1
    \end{cases}
$$

$$
\pmb\beta =  \begin{cases}
    [a_1, a_2] & \mbox{if } x < 1 \\
    [a_3,  a_4] & \mbox{if } x \geq 1
    \end{cases}
$$

$$
\pmb\Phi(x) = [1, x] 
$$



### (c)

$$
\hat{y} = (1+a_1x_1)e^{-x_2+a_2}
$$

$$
\pmb\beta = [e^{a_2}, a_1e^{a_2}]
$$

$$
\pmb\Phi(x_1, x_2) = [e^{-x_2}, x_1e^{-x_2}]
$$

## 4.

### (a)

$$
\pmb\beta = [a_1, a_2, a_3,\dots,a_M, b_0, b_1, b_2, \dots,b_N ]^T
$$

### (b)

$$
\pmb A = \begin{bmatrix} 
0&0&0&\dots&0&x_0&0&0&0&\dots&0\\ 
y_{0}&0&0&\dots&0&x_1&x_0&0&0&\dots&0\\
y_1&y_0&0&\dots&0&x_2&x_1&x_0&0&\dots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\

y_{M-1}&y_{M-2}&\dots&\dots&y_{0}&x_N&x_{N-1}&\dots&\dots&x_1&x_0
\end{bmatrix}
$$



### (c)



## 6.

#### (a)

```python
yhat = beta[0] * X[:, 0] + beta[1] * X[:, 1] + beta[2] * X[:, 1] * X[:, 2]
```

#### (b)

```python
yhat = np.sum(alpha * np.exp(-beta )) * x
```

#### (c)

```python
n,d = x.shape
m,d = y.shape

dist = np.sum(np.square(np.einsum("dm,ln->nmd",y.T, np.ones((1,n))) - np.einsum("ml,nd->nmd", np.ones((m,1)), x)), axis=2)
```



 1,3,4,6

