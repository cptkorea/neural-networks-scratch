import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return np.where(x >= 0, 1, 0)

def mse(x, y):
    return np.mean(np.square(x-y))

def dmse(x, y):
  n = len(x)
  return 2*(y-x) / n

def grad_mse(X, w, y):
    return X.T.dot(X.dot(w) - y)

def train_ls(X, y, alpha=0.05, tol=0.005):
    w = np.random.randn(2)
    grad_loss = grad_mse(X, w, y)
    loss = []
    while np.linalg.norm(grad_loss) >= tol:
        w = w - alpha * grad_loss
        grad_loss = grad_mse(X, w, y)
        loss.append(grad_loss)
    return w

def grad_descent(start, grad_loss, alpha, tol=0.001):
    w = start
    loss = grad_loss(w)
    num_iter = 1
    while loss >= tol:
        w = w - alpha * loss
        num_iter += 1
    return w, num_iter
