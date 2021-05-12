from jax import numpy as jnp

from .. import enum

class Softmax():

	def __init__(self):

		self.types = {enum.ReLU, enum.Activation, enum.NonParametric, enum.ForwardNormalize}
		self.scale = {
				'max': None, 
				'sum': None
			}

	def __call__(self, input_shape):

		self.input_shape = input_shape
		self.output_shape = (None, *input_shape[1:])

	def forward(self, X):

		self.scale['max'] = jnp.max(X, axis = -1, keepdims = True)

		X -= self.scale['max']
		X = jnp.exp(X)

		self.scale['sum'] = jnp.sum(X, axis = -1, keepdims = True)

		X /= self.scale['sum']
		
		return X

	def backward(self, Y):

		Y *= self.scale['sum']
		Y = jnp.log(Y)

		Y = jnp.nan_to_num(Y, neginf = jnp.log(jnp.finfo(jnp.float64).eps), posinf = jnp.log(jnp.finfo(jnp.float32).max))

		Y += self.scale['max']
		
		return Y