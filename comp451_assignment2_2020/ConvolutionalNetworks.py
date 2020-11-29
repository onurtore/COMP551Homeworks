#!/usr/bin/env python
# coding: utf-8

# # Convolutional Networks
# So far we have worked with deep fully-connected networks, using them to explore different optimization strategies and network architectures. Fully-connected networks are a good testbed for experimentation because they are very computationally efficient, but in practice all state-of-the-art results use convolutional networks instead.
# 
# First you will implement several layer types that are used in convolutional networks. You will then use these layers to train a convolutional network on the CIFAR-10 dataset.

# In[1]:


# As usual, a bit of setup
import numpy as np
import matplotlib.pyplot as plt
from comp451.classifiers.cnn import *
from comp451.data_utils import get_CIFAR10_data
from comp451.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from comp451.layers import *
from comp451.fast_layers import *
from comp451.solver import Solver


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)


# # Convolution: Naive forward pass
# The core of a convolutional network is the convolution operation. In the file `comp451/layers.py`,
#  implement the forward pass for the convolution layer in the function `conv_forward_naive`. 
# 
# You don't have to worry too much about efficiency at this point; just write the code in whatever way you find most clear.
# Please do not flip the filter during convolution.
# You can test your implementation by running the following:

# In[ ]:


x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = conv_forward_naive(x, w, b, conv_param)
correct_out = np.array([[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]])

# Compare your output to ours; difference should be around e-8
print('Testing conv_forward_naive')
print('difference: ', rel_error(out, correct_out))


# # Aside: Image processing via convolutions
# 
# As fun way to both check your implementation and gain a better understanding of the type of operation that convolutional layers can perform, we will set up an input containing two images and manually set up filters that perform common image processing operations (grayscale conversion and edge detection). The convolution forward pass will apply these operations to each of the input images. We can then visualize the results as a sanity check.
# 

# In[ ]:



np.random.seed(451)
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# The errors should be around e-8 or less.
print('Testing conv_backward_naive function')
print('dw error: ', rel_error(dw, dw_num))
print('db error: ', rel_error(db, db_num))
print('dx error: ', rel_error(dx, dx_num))


# # Max-Pooling: Naive forward
# Implement the forward pass for the max-pooling operation in the function `max_pool_forward_naive` in the file `comp451/layers.py`. Again, don't worry too much about computational efficiency.
# 
# Check your implementation by running the following:

# In[ ]:


x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# Compare your output with ours. Difference should be on the order of e-8.
print('Testing max_pool_forward_naive function:')
print('difference: ', rel_error(out, correct_out))


# # Max-Pooling: Naive backward
# Implement the backward pass for the max-pooling operation in the function `max_pool_backward_naive` in the file `comp451/layers.py`. You don't need to worry about computational efficiency.
# 
# Check your implementation with numeric gradient checking by running the following:

# In[ ]:


np.random.seed(451)
x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

out, cache = max_pool_forward_naive(x, pool_param)
dx = max_pool_backward_naive(dout, cache)

# Your error should be on the order of e-12
print('Testing max_pool_backward_naive function:')
print('dx error: ', rel_error(dx, dx_num))


# # Average-Pooling: Naive forward
# Implement the forward pass for the average-pooling operation in the function `avg_pool_forward_naive` in the file `comp451/layers.py`. Again, don't worry too much about computational efficiency.
# 
# Check your implementation by running the following:

# In[ ]:


x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = avg_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.28157895, -0.26684211],
                         [-0.22263158, -0.20789474]],

                        [[-0.16368421, -0.14894737],
                         [-0.10473684, -0.09      ]],

                        [[-0.04578947, -0.03105263],
                         [ 0.01315789,  0.02789474]]],


                       [[[ 0.07210526,  0.08684211],
                         [ 0.13105263,  0.14578947]],

                        [[ 0.19      ,  0.20473684],
                         [ 0.24894737,  0.26368421]],

                        [[ 0.30789474,  0.32263158],
                         [ 0.36684211,  0.38157895]]]])

# Compare your output with ours. Difference should be on the order of e-7.
print('Testing avg function:')
print('difference: ', rel_error(out, correct_out))


# # Average-Pooling: Naive backward
# Implement the backward pass for the avg-pooling operation in the function `avg_pool_backward_naive` in the file `comp451/layers.py`. You don't need to worry about computational efficiency.
# 
# Check your implementation with numeric gradient checking by running the following:

# In[ ]:


np.random.seed(451)
x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(lambda x: avg_pool_forward_naive(x, pool_param)[0], x, dout)

out, cache = avg_pool_forward_naive(x, pool_param)
dx = avg_pool_backward_naive(dout, cache)

# Your error should be on the order of e-11
print('Testing avg_pool_backward_naive function:')
print('dx error: ', rel_error(dx, dx_num))


# ## Inline Question 1: 
# 
# Comparing the max-pool and average-pool operations from a theoretical perspective, which one would you choose if you are developing a computer vision application? What is your motivation to choose one over the other?
# 
# ## Answer:
#     

# # Fast layers
# Making convolution and pooling layers fast can be challenging. To spare you the pain, we've provided fast implementations of the forward and backward passes for convolution and pooling layers in the file `comp451/fast_layers.py`.
# 
# The fast convolution implementation depends on a Cython extension; to compile it you need to run the following from the `comp451` directory:
# 
# ```bash
# python setup.py build_ext --inplace
# ```
# 
# The API for the fast versions of the convolution and pooling layers is exactly the same as the naive versions that you implemented above: the forward pass receives data, weights, and parameters and produces outputs and a cache object; the backward pass recieves upstream derivatives and the cache object and produces gradients with respect to the data and weights.
# 
# **NOTE:** The fast implementation for pooling will only perform optimally if the pooling regions are non-overlapping and tile the input. If these conditions are not met then the fast pooling implementation will not be much faster than the naive implementation.
# 
# You can compare the performance of the naive and fast versions of these layers by running the following:

# In[ ]:


# Rel errors should be around e-9 or less
from comp451.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
np.random.seed(451)
x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
t2 = time()

print('Testing conv_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Difference: ', rel_error(out_naive, out_fast))

t0 = time()
dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
t1 = time()
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

print('\nTesting conv_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('dx difference: ', rel_error(dx_naive, dx_fast))
print('dw difference: ', rel_error(dw_naive, dw_fast))
print('db difference: ', rel_error(db_naive, db_fast))


# In[ ]:


# Relative errors should be close to 0.0
from comp451.fast_layers import max_pool_forward_fast, max_pool_backward_fast
np.random.seed(451)
x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
t2 = time()

print('Testing pool_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('difference: ', rel_error(out_naive, out_fast))

t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

print('\nTesting pool_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('dx difference: ', rel_error(dx_naive, dx_fast))


# # Convolutional "sandwich" layers
# Previously we introduced the concept of "sandwich" layers that combine multiple operations into commonly used patterns. In the file `comp451/layer_utils.py` you will find sandwich layers that implement a few commonly used patterns for convolutional networks. Run the cells below to sanity check they're working.

# In[ ]:


# conv - lrelu - pool
from comp451.layer_utils import conv_lrelu_pool_forward, conv_lrelu_pool_backward
np.random.seed(451)
x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
lrelu_param = {'alpha': -2e-3}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_lrelu_pool_forward(x, w, b, conv_param, lrelu_param, pool_param)
dx, dw, db = conv_lrelu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_lrelu_pool_forward(x, w, b, conv_param, lrelu_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_lrelu_pool_forward(x, w, b, conv_param, lrelu_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_lrelu_pool_forward(x, w, b, conv_param, lrelu_param, pool_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing conv_leakyRelu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


# In[ ]:


# conv - lrelu
from comp451.layer_utils import conv_lrelu_forward, conv_lrelu_backward
np.random.seed(451)
x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
lrelu_param = {'alpha': 1e-2}

out, cache = conv_lrelu_forward(x, w, b, conv_param, lrelu_param)
dx, dw, db = conv_lrelu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_lrelu_forward(x, w, b, conv_param, lrelu_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_lrelu_forward(x, w, b, conv_param, lrelu_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_lrelu_forward(x, w, b, conv_param, lrelu_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing conv_leakyRelu:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


# # Three-layer ConvNet
# Now that you have implemented all the necessary layers, we can put them together into a simple convolutional network.
# 
# Open the file `comp451/classifiers/cnn.py` and complete the implementation of the `ThreeLayerConvNet` class. Remember you can use the fast/sandwich layers (already imported for you) in your implementation. Run the following cells to help you debug:

# ## Sanity check loss
# After you build a new network, one of the first things you should do is sanity check the loss. When we use the softmax loss, we expect the loss for random weights (and no regularization) to be about `log(C)` for `C` classes. When we add regularization the loss should go up slightly.

# In[ ]:


model = ThreeLayerConvNet()
print(model.params['W1'].shape)
print(model.params['b1'].shape)
print(model.params['W2'].shape)
print(model.params['b2'].shape)
print(model.params['W3'].shape)
print(model.params['b3'].shape)


# In[ ]:


model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss)

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)


# ## Gradient check
# After the loss looks reasonable, use numeric gradient checking to make sure that your backward pass is correct. When you use numeric gradient checking you should use a small amount of artifical data and a small number of neurons at each layer. Note: correct implementations may still have relative errors up to the order of e-2.

# In[ ]:


num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
np.random.seed(451)
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)
loss, grads = model.loss(X, y)
# Errors should be small, but correct implementations may have
# relative errors up to the order of e-2
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# ## Overfit small data
# A nice trick is to train your model with just a few training samples. You should be able to overfit small datasets, which will result in very high training accuracy and comparatively low validation accuracy.

# In[ ]:


np.random.seed(451)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2)

solver = Solver(model, small_data,
                num_epochs=15, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()


# Plotting the loss, training accuracy, and validation accuracy should show clear overfitting:

# In[ ]:


plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# ## Train the net
# By training the three-layer convolutional network for one epoch, you should achieve greater than 40% accuracy on the training set:

# In[ ]:


model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, alpha=1e-2)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()


# ## Visualize Filters
# You can visualize the first-layer convolutional filters from the trained network by running the following:

# In[ ]:


from comp451.vis_utils import visualize_grid

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()


# In[ ]:




