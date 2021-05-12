from jax import numpy as jnp
from jax.nn import sigmoid

from .. import enum

class Sigmoid():

	def __init__(self):

		self.types = {enum.Sigmoid, enum.Activation, enum.NonParametric}

		self.scale = {
				'min': 0.0,
				'max': 1.0
			}

	def __call__(self, input_shape):

		self.input_shape = input_shape
		self.output_shape = (None, *input_shape[1:])

	def forward(self, X):

		X = sigmoid(X)

		X = X * (self.scale['max'] - self.scale['min']) + self.scale['min']
		
		return X

	def backward(self, Y):

		self.scale['min'], self.scale['max'] = jnp.min(Y), jnp.max(Y)

		Y = (Y - self.scale['min']) / (self.scale['max'] - self.scale['min'])

		Y = jnp.log(Y / (1 - Y))

		# -inf = log(0) ~= log(eps); inf = log(inf) ~= log(max)
		# log(float64.eps) = -36.04365, log(float32.max) = 88.72284
		# log(float32.eps) = -15.94239, log(float16.max) = 11.09
		# log(float16.eps) = -6.93, log(float16.max) = 11.09
		Y = jnp.nan_to_num(Y, neginf = jnp.log(jnp.finfo(jnp.float64).eps), posinf = jnp.log(jnp.finfo(jnp.float32).max))

		return Y
