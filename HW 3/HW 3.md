# Homework3

Lingfeng Zhao

LZ1973

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

For $(1/T)A^TA​$, it is easy to get the result that:
$$
(1/T)A^TA = \begin{bmatrix}
R_{yy}(0)&R_{yy}(1)&R_{yy}(2)&\dots& R_{yy}(M-1)&R_{xy}(-1)&R_{xy}(0)&R_{xy}(1)&\dots& R_{xy}(N-1) \\
R_{yy}(1)&R_{yy}(0)&R_{yy}(1)&\dots& R_{yy}(M-2)&R_{xy}(-2)&R_{xy}(-1)&R_{xy}(-2)&\dots& R_{xy}(N-2) \\
R_{yy}(2)&R_{yy}(1)&R_{yy}(0)&\dots& R_{yy}(M-3)&R_{xy}(-3)&R_{xy}(-2)&R_{xy}(-1)&\dots& R_{xy}(N-3) \\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\ddots&\vdots \\
R_{xy}(N-1)&R_{xy}(N-2)&R_{xy}(N-3)&\dots&R_{xy}(-1)&R_{xx}(N)&R_{xx}(N-1)&\dots&\dots&R_{xx}(0)

\end{bmatrix}
$$
It is, for element $m_{ij}​$ in $(1/T)A^TA​$ 
$$
m{ij} = \begin{cases}
R_{yy}(|{i-j}|) & \mbox{if } 0\le i,j\leq M\\
R_{xy}(i - |j - M|) & \mbox{if } 0\le i \le M \text{ and } j>M \\
               & \text{or }     0\le j\le M \text{ and } i>M   \\
R_{xx}(|{i-j}|) & \mbox{if } M < i,j\\
\end{cases}
$$
For $(1/T)A^Ty$
$$
(1/T)A^Ty = [ R_{yy}(1),R_{yy}(2),R_{yy}(3),\dots,R_{yy}(M-1),R_{xy}(0),R_{xy}(1),\dots,R_{xy}(N) ]^T
$$

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

