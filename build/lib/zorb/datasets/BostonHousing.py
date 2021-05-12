from numpy import reshape
from tensorflow.keras.datasets import boston_housing

class BostonHousing():

	def __init__(self, train_samples = None, test_samples = None):

		train, test = boston_housing.load_data()

		self.train_x, self.train_y = train
		self.test_x, self.test_y = test

		self.train_y = reshape(self.train_y, (-1, 1))
		self.test_y = reshape(self.test_y, (-1, 1))

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