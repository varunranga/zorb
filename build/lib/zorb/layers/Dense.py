from jax import numpy as jnp
from jax import random
from jax.nn import initializers

from .. import enum
from .. import key

class Dense():

	def __init__(self, units, include_bias = True, initializer = "glorot_uniform", rcond = None, u_rcond = None, b_rcond = None):

		self.types = {enum.Dense, enum.Parametric}
		self.units = units
		self.include_bias = include_bias
		self.initializer = initializer

		self.u_rcond = u_rcond if u_rcond else rcond
		self.b_rcond = b_rcond if b_rcond else rcond

		self.output_shape = (None, self.units)

	def __call__(self, input_shape):

		self.input_shape = input_shape

		initializer = eval("initializers." + self.initializer)
		self.W = initializer()(key, shape = (input_shape[1] + (1 if self.include_bias else 0), self.units))

	def forward(self, X):

		if self.include_bias: X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis = 1)

		H = jnp.matmul(X, self.W)

		return H

	def backward(self, Y):	

		if self.include_bias:

			Y_minus_B = jnp.subtract(Y, self.W[-1])

			pinv_W = jnp.linalg.pinv(self.W[:-1], rcond = self.b_rcond)

			F = jnp.matmul(Y_minus_B, pinv_W)

		else:

			pinv_W = jnp.linalg.pinv(self.W, rcond = self.b_rcond)

			F = jnp.matmul(Y, pinv_W)

		return F

	def update(self, X, Y):

		if self.include_bias: X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis = 1)

		pinv_X = jnp.linalg.pinv(X, rcond = self.u_rcond)

		self.W = jnp.matmul(pinv_X, Y)

		return True if not (jnp.isinf(self.W).any() or jnp.isnan(self.W).any()) else False
