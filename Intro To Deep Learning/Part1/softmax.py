import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    f = X.dot(W)
    (N,C) = f.shape
    lossArray = np.zeros(f.shape[0])

    #Looping over Individual Image Class results
    for i in range(N):
        f[i] -= np.max(f[i]) #Done for Numerical Stability
        f[i] = np.exp(f[i]) #Take exponent of whole f matrix

        #Softmax Function
        lossArray[i] = f[i][y[i]] / np.sum(f[i])

        #Computing Gradients
        for j in range(C):
            p = f[i][j]/np.sum(f[i])
            dW[:, j] += (p-(j == y[i])) * X[i, :]

    #Cross-Entropy Loss
    loss = np.mean(-np.log(lossArray))
    reg_loss = 0.5*reg*np.sum(W*W)
    loss+=reg_loss

    dW /= N
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    a = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    f = np.dot(X,W)
    (N,C) = f.shape
    f -= np.max(f)
    
    y_mat = np.zeros_like(f)
    y_mat[range(N),y] = 1

    req_f = np.multiply(y_mat, f)
    sum_f = np.sum(req_f,axis=1)

    exp_f = np.exp(f)
    sum_exp_f = np.sum(exp_f,axis=1)

    loss = np.mean(-np.log(np.divide(np.exp(sum_f),sum_exp_f)))
    reg_loss = 0.5*reg*np.sum(W*W)
    loss+=reg_loss

    dW = np.divide(exp_f.T,sum_exp_f)
    dW = np.dot(dW,X)
    dW -= np.dot(y_mat.T,X)

    dW /= N
    dW = dW.T
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

