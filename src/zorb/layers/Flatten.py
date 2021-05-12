import jax.numpy as jnp

from collections.abc import Iterable

from .. import enum
from .. import np

class Flatten():

	def __init__(self):

		self.types = {enum.Flatten, enum.NonParametric}

	def __call__(self, input_shape):

		self.input_shape = input_shape
		self.output_shape = (None, np.prod(np.array(input_shape[1:])))

	def forward(self, x):

		x = jnp.reshape(x, (-1, self.output_shape[1]))

		return x

	def backward(self, y):

		y = jnp.reshape(y, (-1,) + self.input_shape[1:])
			
		return y
