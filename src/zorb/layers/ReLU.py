from jax import numpy as jnp
from jax.nn import relu
from jax.lax import cond

from .. import enum

class ReLU():

	def __init__(self):

		self.types = {enum.ReLU, enum.Activation, enum.NonParametric}

	def __call__(self, input_shape):

		self.input_shape = input_shape
		self.output_shape = (None, *input_shape[1:])

	def forward(self, X): return relu(X)

	def backward(self, Y): return relu(Y)
