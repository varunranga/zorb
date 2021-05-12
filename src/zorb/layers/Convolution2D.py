from jax import numpy as jnp
from numpy import zeros

from .Dense import Dense
from .. import enum

class Convolution2D():

	def __init__(self, n_kernels, size, stride = 1, include_bias = True, initializer = "glorot_uniform", rcond = None, u_rcond = None, b_rcond = None):

		self.types = {enum.Convolution2D, enum.Parametric}
		self.n_kernels = n_kernels
		self.size = size
		self.stride = stride

		self.dense = Dense(units = self.n_kernels, include_bias = include_bias, initializer = initializer, rcond = rcond, u_rcond = u_rcond, b_rcond = b_rcond)
		
	def __call__(self, input_shape):

		self.input_shape = input_shape

		self.channels = self.input_shape[-1]

		_i = 0
		for i in range(0, self.input_shape[1], self.stride):
			if (i+self.size > input_shape[1]): break
			_i += 1

		_j = 0
		for j in range(0, input_shape[2], self.stride):
			if (j+self.size > input_shape[2]): break
			_j += 1

		self.output_shape = (None, _i, _j, self.n_kernels)

		self.dense(input_shape = (None, self.size**2 * self.channels))

		self.W = self.dense.W

	def forward(self, X):

		_X = zeros((X.shape[0] * self.output_shape[1] * self.output_shape[2], self.size**2 * self.channels), dtype = 'float16')
		idx = jnp.array(list(map(lambda i: list(range(i, _X.shape[0], self.output_shape[1] * self.output_shape[2])), range(self.output_shape[1] * self.output_shape[2]))))	

		c = 0
		for i in range(0, X.shape[1], self.stride): 
			for j in range(0, X.shape[2], self.stride): 
				if (i+self.size <= X.shape[1]) and (j+self.size <= X.shape[2]):
					_X[idx[c]] = jnp.reshape(X[:, i:i+self.size, j:j+self.size], (X.shape[0], -1))
					c += 1

		_X = self.dense.forward(_X)

		X = jnp.reshape(_X, (X.shape[0],) + self.output_shape[1:])

		return X

	def backward(self, Y):

		Y = jnp.reshape(Y, (-1, self.n_kernels))

		Y = self.dense.backward(Y)

		Y = jnp.reshape(Y, (-1, self.output_shape[1], self.output_shape[2], self.size**2 * self.channels)) 

		_Y = zeros((Y.shape[0], *self.input_shape[1:]), dtype = 'float32')
		counts = zeros((Y.shape[0], *self.input_shape[1:]), dtype = 'int16')

		for i in range(Y.shape[1]):
			for j in range(Y.shape[2]):
				_Y[:, i:i+self.size, j:j+self.size] += jnp.reshape(Y[:, i, j], (-1, self.size, self.size, self.channels))
				counts[:, i:i+self.size, j:j+self.size] += 1

		Y = _Y / counts

		return Y

	def update(self, X, Y):

		_X = zeros((X.shape[0] * self.output_shape[1] * self.output_shape[2], self.size**2 * self.channels), dtype = 'float16')
		idx = jnp.array(list(map(lambda i: list(range(i, _X.shape[0], self.output_shape[1] * self.output_shape[2])), range(self.output_shape[1] * self.output_shape[2]))))	

		c = 0
		for i in range(0, X.shape[1], self.stride): 
			for j in range(0, X.shape[2], self.stride): 
				if (i+self.size <= X.shape[1]) and (j+self.size <= X.shape[2]):
					_X[idx[c]] = jnp.reshape(X[:, i:i+self.size, j:j+self.size], (X.shape[0], -1))
					c += 1

		Y = jnp.reshape(Y, (-1, self.n_kernels))

		self.dense.update(_X, Y)
