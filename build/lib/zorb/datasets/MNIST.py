from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from numpy import random
from numpy import expand_dims

class MNIST():

	def __init__(self, train_samples = None, test_samples = None):

		train, test = mnist.load_data()

		self.train_x, self.train_y = train
		self.test_x, self.test_y = test

		self.train_x = expand_dims(self.train_x, -1)
		self.test_x = expand_dims(self.test_x, -1)

		self.train_y = to_categorical(self.train_y)
		self.test_y = to_categorical(self.test_y)

		if train_samples:

			if isinstance(train_samples, int):
				samples = random.randint(low = 0, high = len(self.train_x), size = train_samples)
			elif isinstance(train_samples, float):
				samples = random.randint(low = 0, high = len(self.train_x), size = int(train_samples * len(self.train_x)))

			self.train_x = self.train_x[samples]
			self.train_y = self.train_y[samples]

		if test_samples:

			if isinstance(test_samples, int):
				samples = random.randint(low = 0, high = len(self.test_x), size = test_samples)
			elif isinstance(test_samples, float):
				samples = random.randint(low = 0, high = len(self.test_x), size = int(test_samples * len(self.test_x)))

			self.test_x = self.test_x[samples]
			self.test_y = self.test_y[samples]

		self.dimensions = {'input': {'train': self.train_x.shape, 'test': self.test_x.shape}, 
					   	   'output': {'train': self.train_y.shape, 'test': self.test_y.shape}}