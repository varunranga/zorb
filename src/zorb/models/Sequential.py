from jax import numpy as jnp

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

from collections.abc import Iterable

from .. import enum
from .. import seed

# from ..layers import *

class Sequential():

	def __init__(self, input_shape, network = None):
		
		self.input_shape = (None, *input_shape)
		self._last_outgoing_shape = self.input_shape

		self.layers = []
		self.number_of_layers = 0
		self._config_changed = True

		self.add(network)

	def add(self, network):

		if isinstance(network, Iterable):
	
			for layer in network:
			
				self.layers.append(layer)

				self.layers[-1](input_shape = self._last_outgoing_shape)
				
				self._last_outgoing_shape = self.layers[-1].output_shape

		else:
	
			self.layers.append(network)

			if enum.Parametric in self.layers[-1].types:
				self._last_outgoing_shape = self.layers[-1].output_shape

		self.number_of_layers = len(self.layers)

		self._config_changed = True

	def summary(self):

		dct = {}
		total_parameters = 0

		print("Seed:", seed, end = "\n\n")
		print("Sequential")
		print("----------\n")

		print('\t', "Input:")
		print('\t', '\t', "Output Shape:", self.input_shape, '\n')

		for layer in self.layers:
			object_name = layer.__str__().split(' ')[0].split('.')[-1]
			parameters = jnp.prod(jnp.array(layer.W.shape)) if enum.Parametric in layer.types else 0

			if object_name not in dct: dct[object_name] = 0

			print('\t', object_name+str(dct[object_name]), ':')
			print('\t', '\t', "Output Shape:", layer.output_shape)
			print('\t', '\t', "# of parameters:", parameters, '\n')

			total_parameters += parameters
			dct[object_name] += 1

		print("Total # of parameters", total_parameters, '\n')

	def fit(self, X, Y):

		saved_Y = Y

		for i in range(self.number_of_layers):

			saved_X = X

			if enum.Parametric in self.layers[i].types:
	
				Y = saved_Y

				for layer in self.layers[i:]: X = layer.forward(X)
				for layer in self.layers[:i:-1]: Y = layer.backward(Y)

				self.layers[i].update(saved_X, Y)

			X = self.layers[i].forward(saved_X)

	def predict(self, X):
		
		for layer in self.layers: X = layer.forward(X)

		return X

	def evaluate(self, X, Y):

		predict_Y = self.predict(X)

		result = {}

		result['mean_squared_error'] = mean_squared_error(Y, predict_Y)
		result['log_loss'] = log_loss(Y, predict_Y)

		if self._last_outgoing_shape[-1] == 1:

			predict_Y = jnp.round(jnp.ravel(predict_Y))
			Y = jnp.round(jnp.ravel(Y))

		elif self._last_outgoing_shape[-1] > 1:

			predict_Y = jnp.argmax(predict_Y, axis = 1)
			Y = jnp.argmax(Y, axis = 1)

		result['accuracy_score'] = accuracy_score(Y, predict_Y)

		return result
