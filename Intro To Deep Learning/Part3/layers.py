import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_ = 1 + (H + 2*pad - HH)/stride
    W_ = 1 + (W + 2*pad - WW)/stride

    x_ = np.pad(x,((0,0),(0,0),(int(pad),int(pad)),(int(pad),int(pad))),'constant')
    out = np.zeros((N,F,int(H_),int(W_)))
    for i in range(N):
      for j in range(F):
        for idxK, k in enumerate(range(0,H-HH+1+2*pad,stride)):
          for idxL, l in enumerate(range(0,W-WW+1+2*pad,stride)):
            out[i][j][idxK][idxL] = np.dot(np.reshape(x_[i][:][k:k+HH][l:l+WW],(C*HH*WW)),np.reshape(w[j],(C*HH*WW)))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    print(out)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################

    '''
    N, C, H, W = x.shape
    newheight = W/pool_param['pool_height']
    newWidth = W/pool_param['pool_width']
    stride = pool_param['stride']
    out = np.zeros((N,C,int(newheight),int(newWidth)))

    for i in range(N):
      for j in range(C):
        for idxK, k in enumerate(range(0,H,stride)):
          for idxL, l in enumerate(range(0,W,stride)):
            print(k)
            print(pool_param['pool_height'])
            print(l)
            print(pool_param['pool_width'])
            print(x[i][j][k:k+pool_param['pool_height']][l:l+pool_param['pool_width']])
            print(x[0][0][0:2][0:2])
            out[i][j][idxK][idxL] = np.amax(2)

    '''
    (N, C, H, W) = x.shape
    maxIdx = np.amax([N,C,H,W,2])
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + (H - pool_height) / stride
    W_prime = 1 + (W - pool_width) / stride

    out = np.zeros((N, C, int(H_prime), int(W_prime)))

    for n in range(N):
      for h in range(int(H_prime)):
        for w in range(int(W_prime)):
          h1 = h * stride
          h2 = h * stride + pool_height
          w1 = w * stride
          w2 = w * stride + pool_width
          window = x[n, :, h1:h2, w1:w2]
          out[n,:,h,w] = np.max(window.reshape((C, pool_height*pool_width)), axis=1)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################

    x, maxIdx, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + (H - pool_height) / stride
    W_prime = 1 + (W - pool_width) / stride

    dx = np.zeros_like(x)

    for n in range(N):
      for c in range(C):
        for h in range(int(H_prime)):
          for w in range(int(W_prime)):
            h1 = h * stride
            h2 = h * stride + pool_height
            w1 = w * stride
            w2 = w * stride + pool_width
            window = x[n, c, h1:h2, w1:w2]
            window2 = np.reshape(window, (pool_height*pool_width))
            window3 = np.zeros_like(window2)
            window3[np.argmax(window2)] = 1

            dx[n,c,h1:h2,w1:w2] = np.reshape(window3,(pool_height,pool_width)) * dout[n,c,h,w]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx
