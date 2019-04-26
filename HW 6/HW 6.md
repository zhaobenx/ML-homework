# Homework 6

Lingfeng Zhao

LZ1973

## 2

### (a)

Note that $y = 0, 1$, hence
$$
P(y =0|\mathbf{x}) = 1 - P(y = 1|\mathbf{x}) = 1 - \frac{1}{1+e^{-z}} = \frac{e^{-z}}{1+e^{-z}}
$$
For $P(y =1|\mathbf{x})\gt P(y =0|\mathbf{x})$, we get
$$
\begin{align}
     &\frac{1}{1+e^{-z}}-\frac{e^{-z}}{1+e^{-z}} > 0 \\
     \Rightarrow\ &\frac{1- e^{-z}}{1+e^{-z}}\gt0\\
     \Rightarrow\ &1 - e^{-z} > 0 \\
     \Rightarrow\ &z > 0 \\
     \Rightarrow\ &\beta_0 +\beta_1x_1+\beta_2x_2>0
     
 \end{align}
$$

### (b)

In this case
$$
\begin{align}
     &P(y =1|\mathbf{x}) > 0.8 \\
     \Rightarrow\ &e^{-z} < 0.25 \\
     \Rightarrow\ &e>\ln{4} \\
     \Rightarrow\ &\beta_0 +\beta_1x_1+\beta_2x_2> \ln{4}
     
 \end{align}
$$


### （c)

Given $P(y =1|\mathbf{x}) > 0.8 $ and $x_2 = 0.5$. 
$$
\begin{align}
     &\beta_0 +\beta_1x_1+\beta_2x_2> \ln{4} \\
     \Rightarrow\ &\beta_0 +\beta_1x_1+0.5\beta_2> \ln{4} \\
     \Rightarrow\ &x_1 > ln2 - 1.25\\
 \end{align}
$$

## 3

### (a)

![figure1](D:\Documents\NYU\2019\ML\homework\HW 6\Figure_1.svg)

### (b)

We can choose $x_2 = 0.5 $ as the classifier. So the $z_i = \mathbf {w^Tx}_i + b$ , so we can define $\mathbf w = [0, 2]^T$, $b = -1$.

### (c)

| Income (thousands $), xi1  | 30 |50 |70 |80| 100       |
| -------------------------- | --------------- | ---- | ---- | ---- | ---- |
| Num websites, xi2          | 0| 1 |1 |2 |1 |
| Donate (1=yes or 0=no), yi | 0 |1 |0 |1 |1       |
| $z_i = 2x_2 -1$ | -1 |1 |1 |3 |1 |
| $P(y_i |\mathbf{x_i})$ | 0.7311 | 0.7311 |0.2689 | 0.9526 | 0.7311|

We can see when $i=3$ the probability is the smallest.

### (d)

It will change the probability, as the $z_i$ is changed. The new $z'_i = \alpha z_i$, it is obvious that if $\alpha > 1$, the probability will increase; if $\alpha < 1$ the probability will decrease.

## 4

### (a)

$$
z = \beta_0+\beta_1X_1+\beta_2X_2 = -6 + 0.05\times40 +3.5 = -0.5\\
P(Y=1|X=[40,3.5]) = \frac{1}{1+e^{-z}} = 0.3775
$$

### (b)

To get 50% chance of getting A
$$
\frac{1}{1+e^{-z}} > 0.5 \Rightarrow z > 0 \\
-6 +0.05\times X_1 +3.5 > 0 \Rightarrow X_1 > 50
$$


So the student should spend at least 50 hours to get an A.