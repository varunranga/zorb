from jax import numpy as jnp
from jax.lax import tanh
from jax.lax import atanh

from .. import enum

class Tanh():

	def __init__(self):

		self.types = {enum.Tanh, enum.Activation, enum.NonParametric}

		self.scale = {
				'min': -1.0,
				'max': 1.0
			}

	def __call__(self, input_shape):

		self.input_shape = input_shape
		self.output_shape = (None, *input_shape[1:])

	def forward(self, X):

		X = tanh(X)

		X = ((X - (-1)) * (self.scale['max'] - self.scale['min']) / (1 - (-1))) + self.scale['min']

		return X

	def backward(self, Y):

		self.scale['min'], self.scale['max'] = jnp.min(Y), jnp.max(Y)

		Y = (Y - self.scale['min']) * (1 - (-1)) / (self.scale['max'] - self.scale['min']) + (-1)

		Y = atanh(Y)

		Y = jnp.nan_to_num(Y, neginf = jnp.log(jnp.finfo(jnp.float64).eps), posinf = jnp.log(jnp.finfo(jnp.float32).max))

		return Y
