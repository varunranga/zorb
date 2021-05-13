# zorb - Zeroth Order Relaxed Backpropagation

This is a keras-style implementation of the Zeroth Order Relaxed Backpropagation ([ZORB](arxiv.org/abs/2011.08895)) on Jax. For reproducible experiments, please refer to the numpy version ([zorb-numpy](https://github.com/varunranga/zorb-numpy)).

ZORB is currently a neural-network training algorithm, which can be extended for a general computational graph.
Similar to how the Backpropation Algorithm extends Gradient Descent for neural networks, ZORB extends the [linear least squares](https://en.wikipedia.org/wiki/Linear_least_squares) method to a non-linear neural network function. 

## Why should you use ZORB?


1.   **Speed**: ZORB is significantly faster than Adam. No gradient calculations; Single update to variables.
2.   **Accuracy**: ZORB achieves comparable accuracies to Adam. Training initial layers (layers closer to the input) first allows better automatic feature extraction.
3.   **No Hyperparameters**: ZORB does away with all hyperparameters that significantly affect training.
4.   **No Objective Functions**: ZORB implicitly reduces the mean squared error.
5.   **Allows non-differentiable layers**: Since ZORB does not calculate any derivatives/gradients, it is compatible with non-differentiable layers as well.

*If the size of the dataset is small to mid-sized (10s - 10Ks), and you wish to train a small yet deep neural network, ZORB will be perfect for you.*

## Where does ZORB lack?

1.   **Space**: ZORB uses the SVD to calculate the pseudo-inverse of matrices. This operation is space exhaustive. ZORB may not be suitable for large datasets.
2.   **Batch Training Only**: ZORB trains neural networks with all data at once. This influences the space complexity as well.
3.   **Supervised Learning Only**: ZORB is currently designed for Supervised Learning tasks only.
4.   **Regularization**: Regularization can be controlled through the ```rcond```, ```u_rcond``` (update rcond), ```b_rcond``` (backprop rcond); currently ```rcond(M) = 10. * max(*M.shape) * jnp.finfo(jnp.float64).eps``` is used to cut-off lower singular values. Very high singular values are not handled, and therefore this package require parameters in ```jnp.float64```.
5.   **Convolution**: Convolution operation is time and space exhaustive. It would be suggested to use only 1 convolution layer, followed by Dense layers.

*If the size of the dataset is large (>100K) or if the Supervised Learning task is requires batching, ZORB might not yield any advantage.*

## Quickstart

The quickest way to start using ZORB is through a docker container:

```
$ docker pull varunranga/zorb
$ docker run  varunranga/zorb \
                      		--dataset MNIST \
                      		--network Convolution2D[8,3] \
					  Sigmoid \
                               		  Flatten \
                                	  Dense[1024] \
                                	  Sigmoid \ 
                                	  Dense[10] \
                                	  Softmax
```

Alternatively, you can install this package by executing these commands:

```
$ pip install -i https://test.pypi.org/simple/ zorbjax==0.0.1
$ echo "alias zorb='<PATH TO zorbjax PACKAGE>/zorb.sh'"
```

You can access ZORB using ```zorb --help```. To create a simple convolution network for MNIST, you may use this command:
```
$ zorb \
         --dataset MNIST \
         --network Convolution2D[8,3] \
	 	   Sigmoid \
                   Flatten \
                   Dense[1024] \
                   Sigmoid \ 
                   Dense[10] \ 
                   Softmax
```

## Getting started with code

This package follows [tensorflow.keras.models.Sequential](https://keras.io/guides/sequential_model/) style of defining neural network. The following code block shows a program written to train a simple convolution network on MNIST.

```
import zorb

# create dataset
dataset = zorb.datasets.MNIST()

# define network
network = [
  # define Convolution2D layers
  zorb.layers.Convolution2D(n_kernels = 8, size = 3, stride = 1),
  zorb.layers.Sigmoid(),

  # flatten for Dense layers
  zorb.layers.Flatten(),

  # define Dense layers
  zorb.layers.Dense(units = 1024), 
  zorb.layers.Sigmoid(),
  zorb.layers.Dense(units = 10), 
  zorb.layers.Softmax()
]

# create model
model = zorb.models.Sequential(input_shape = dataset.dimensions['input']['train'][1:], network = network)

# train model
model.fit(X = dataset.train_x, Y = dataset.train_y)

# evaluate model
print(model.evaluate(X = dataset.test_x, Y = dataset.test_y))
```

## Adding new layers

To add a new layer, simply create a new file ```MyLayer.py``` in the ```src/zorb/layers``` directory with the following definitions:

```
#import required jax functions

# need enum for assigning types to this layer
from .. import enum

class MyLayer():

  def __init__(self, <arguments>):
    self.types = {enum.MyLayer, enum.Parametric} # see src/__init__.py for pre-defined enums
    ...

  def __call__(self, input_shape):
    self.input_shape = input_shape
   
    ... # initialize required parameters
    
    self.output_shape = ... # compute output shape

  def forward(self, X):
    ... # do forward computation
    return H
  
  def backward(self, Y):
    ... # do backward computation
    return F
  
  def update(self, X, Y): # optional function for layer type enum.NonParametric
    ... # do forward computation
    ... # do backward computation

    ... # update rule

    return True if not (jnp.isinf(self.W).any() or jnp.isnan(self.W).any()) else False
```

Add ```enum.MyLayer``` to ```src/zorb/__init__.py```. \\
Add line ```from .MyLayer import MyLayer``` to ```src/zorb/layers/__init__.py```

## Adding new models

Similar to adding a new layer, to create a new model (would involves changes to the algorithm), simply create a new file ```MyModel.py``` in the ```src/zorb/models``` directory with the following definitions:

```
#import required jax functions
#import metric functions

# need enum for assigning types to this layer
from .. import enum

class MyModel():

  def __init__(self, input_shape, network = None):
		
    self.input_shape = (None, *input_shape)
    self._last_outgoing_shape = self.input_shape

    self.layers = []
    self.number_of_layers = 0
    self._config_changed = True

    self.add(network)

  def add(self, network):
    ... # add layer(s) to compute graph

    self.number_of_layers = len(self.layers)
    self._config_changed = True

  def summary(self):
    ... # print a summary of the graph
  
  def fit(self, X, Y):
    ... # new training procedure goes here
  
  def predict(self, X):
    ... # predict output
  
  def evaluate(self, X, Y):
    predicted_Y = self.predict(X)

    ... # evaluate and return dictionary of results
```

Add line ```from .MyModel import MyModel``` to ```src/zorb/models/__init__.py```

## Citing this work

If you use this repository for research purposes, please cite this as follows:

```
@inproceedings{ranganathan2020zorb,
  author    = {Ranganathan, Varun and Lewandowski, Alex},
  title     = {ZORB: A Derivative-Free Backpropagation Algorithm for Neural Networks},
  year      = {2020},
  maintitle = {Neural Information Processing Systems (NeurIPS 2020)},
  booktitle = {Beyond Backpropagation Workshop at NeurIPS 2020},
  url       = {https://arxiv.org/abs/2011.08895},
}
```
