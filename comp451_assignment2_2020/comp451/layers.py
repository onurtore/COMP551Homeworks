from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X   = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    out = np.dot(X,w) + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****    
    X  = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    dx = np.dot(dout,w.T)
    dw = np.dot(X.T,dout)
    db = np.sum(dout,axis=0)
    dx = dx.reshape(x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the Sigmoid forward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = 1 / (1 + np.exp(-x))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoid units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the Sigmoid backward pass.                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    sigmoid = 1 / (1 + np.exp(-x))
    dx = dout * np.multiply(sigmoid,(1-sigmoid))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx  = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def leaky_relu_forward(x, lrelu_param):
    """
    Computes the forward pass for a layer of leaky rectified linear units (Leaky ReLUs).

    Input:
    - x: Inputs, of any shape
    - lrelu_param: Dictionary with the following key:
        - alpha: scalar value for negative slope

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: (x, lrelu_param).
            Input x, of same shape as dout,
            lrelu_param, needed for backward pass.
    """
    out = None
    alpha = lrelu_param.get('alpha', 2e-3)
    ###########################################################################
    # TODO: Implement the Leaky ReLU forward pass.                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(alpha*x,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, lrelu_param)
    return out, cache


def leaky_relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of leaky rectified linear units (Leaky ReLUs).
    Note that, the negative slope parameter (i.e. alpha) is fixed in this implementation.
    Therefore, you should not calculate any gradient for alpha.
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: (x, lr_param)

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    (x, lr_param) = cache
    alpha = lr_param["alpha"]
    ###########################################################################
    # TODO: Implement the Leaky ReLU backward pass.                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x >= 0)
    dx += dout * alpha * (x<0) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) /p
        out  = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def next_window(matrix_shape,kernel_size,stride):
  """
  Given a matrix it returns the window coordinates and output coordinates
  If no more window left then it returns -1

  Please refer to conv_forward_naive function to more information
  about the params
  """

  window_x     = 0
  coordinate_x = 0
  while (window_x + kernel_size[0]) <= matrix_shape[0]:
    window_y     = 0
    coordinate_y = 0
    while (window_y+kernel_size[1]) <= matrix_shape[1]:
      yield (window_x,window_y,coordinate_x,coordinate_y)
      window_y     += stride
      coordinate_y += 1
    window_x     += stride
    coordinate_x += 1
  return -1


def conv(x,w,b):
  """
  Applies a convolution with two windows over all samples, filters, channels
  Please refer to conv_forward_naive function for param info
  """

  res = []
  for i in range(x.shape[0]): #Over samples
    for k in range(w.shape[0]): #Over filters
      out  = 0
      for j in range(x.shape[1]): #Over channels
        out += float(np.dot(x[i,j,:,:].flatten()[:,np.newaxis].T,w[k,j,:,:].flatten()[:,np.newaxis]))
      out += b[k]
      res.append(out)
  return np.array(res).reshape((x.shape[0],w.shape[0]))

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    Note that, the filter is not flipped as in the regular convolution operation
    in signal processing domain. Therefore, technically this implementation
    is a cross-correlation.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_  = np.pad(x, ((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad']) ), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
    H_  = int(1 + (x_.shape[2] * conv_param['pad'] - w.shape[2]) / conv_param['stride'])
    W_  = int(1 + (x_.shape[3] * conv_param['pad'] - w.shape[3]) / conv_param['stride'])
    out = np.zeros((x.shape[0],w.shape[0],H_,W_))#Samples,Filter,H_,W_
    for window in next_window(x_.shape[2:],w.shape[2:],conv_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      out[:,:,start_x,start_y] = conv(x_[:,:,window_x:window_x+w.shape[2],window_y:window_y+w.shape[3]],w,b)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def grad_conv_x(w,dout):

  samples = []
  for j in range(dout.shape[0]):#Over samples
    derivative = np.zeros((w.shape[1], w.shape[2],w.shape[3]))
    for i in range(dout.shape[1]):#Over filters
      derivative += np.multiply(dout[j,i],w[i,:,:,:])
    samples.append(derivative)
  return np.array(samples).reshape((dout.shape[0], w.shape[1],w.shape[2],w.shape[3]))


def grad_conv_b(dout):
  return dout.sum(axis=0)

def grad_conv_w(x,dout):
  """
  Gradient operation for the weight vector of convolution.

  Please refer to conv_forward_naive function to more information
  about the params
  """
  filters = []
  for i in range(dout.shape[1]): #Over filters
    derivative = np.zeros((x.shape[1],x.shape[2],x.shape[3]))
    for j in range(dout.shape[0]): #Over samples
      derivative += np.multiply(dout[j,i],x[j,:,:,:]) #C-H-W
    filters.append(derivative)
  return np.array(filters).reshape((dout.shape[1], x.shape[1],x.shape[2],x.shape[3]))


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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,w,b,conv_param = cache
    x_ = np.pad(x, ((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad']) ), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
    dx = np.zeros(x_.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    for window in next_window(x_.shape[2:],w.shape[2:],conv_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      dw += grad_conv_w(x_[:,:,window_x:window_x+w.shape[2],window_y:window_y+w.shape[3]], dout[:,:,start_x,start_y])
      db += grad_conv_b(dout[:,:,start_x,start_y])
      dx[:,:,window_x:window_x+w.shape[2],window_y:window_y+w.shape[3]] += grad_conv_x(w,dout[:,:,start_x,start_y])
    dx = dx[:,:,conv_param['pad']:-conv_param['pad'], conv_param['pad']:-conv_param['pad']]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def pool(x,type):
  if type == 'max':
    return np.max(x,axis=(2,3))
  if type == 'avg':
    return np.average(x,axis=(2,3))

def grad_pool(x,type,dout):
  if type == 'max':
    idx  = x.reshape(x.shape[0],x.shape[1],-1).argmax(2)
    out  = np.unravel_index(idx,(x.shape[-2:]))
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        grad[i,j,out[0][i,j],out[1][i,j]] = dout[i,j]
    return grad
  if type == 'avg':
    grad = np.zeros(x.shape) #Samples, Channels, H, W
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        grad[i,j] = dout[i,j] / (x.shape[2] * x.shape[3]) 
    return grad

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    H_  = int(1 + (x.shape[2] - pool_param['pool_height']) / pool_param['stride'])
    W_  = int(1 + (x.shape[3] - pool_param['pool_width']) / pool_param['stride'])
    out = np.zeros((x.shape[0],x.shape[1],H_,W_))#Samples,Channels,H_,W_
    for window in next_window(x.shape[2:],(pool_param['pool_height'],pool_param['pool_width']),pool_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      out[:,:,start_x,start_y] = pool(x[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']],'max')


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,pool_param = cache
    dx = np.zeros(x.shape)
    for window in next_window(x.shape[2:],(pool_param['pool_height'],pool_param['pool_width']),pool_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      dx[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']] += grad_pool(x[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']],'max', dout[:,:,start_x,start_y])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def avg_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a avg-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the avg-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    H_  = int(1 + (x.shape[2] - pool_param['pool_height']) / pool_param['stride'])
    W_  = int(1 + (x.shape[3] - pool_param['pool_width']) / pool_param['stride'])
    out = np.zeros((x.shape[0],x.shape[1],H_,W_))#Samples,Channels,H_,W_
    for window in next_window(x.shape[2:],(pool_param['pool_height'],pool_param['pool_width']),pool_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      out[:,:,start_x,start_y] = pool(x[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']],'avg')


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def avg_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a avg-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the avg-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,pool_param = cache
    dx = np.zeros(x.shape)
    for window in next_window(x.shape[2:],(pool_param['pool_height'],pool_param['pool_width']),pool_param['stride']):
      if window == -1:
        break
      window_x,window_y,start_x,start_y = window
      dx[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']] += grad_pool(x[:,:,window_x:window_x+pool_param['pool_height'],window_y:window_y+pool_param['pool_width']],'avg', dout[:,:,start_x,start_y])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx




def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
