from pandas import read_csv
from numpy import stack
from numpy import array
from sklearn.model_selection import train_test_split

class Iris():

	def __init__(self, train_samples = None, test_samples = None):


		labels = {'Iris-versicolor': [1, 0, 0], 'Iris-virginica': [0, 1, 0], 'Iris-setosa': [0, 0, 1]}

		df = read_csv('./datasets/Iris/iris.csv', header = None, sep = ',')

		self.all_x = stack([df[0], df[1], df[2], df[3]], axis = 1)
		self.all_y = array(list(map(lambda x: labels[x], df[4])))

		if train_samples:

			if isinstance(train_samples, int):
				self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.all_x, self.all_y, train_size = train_samples / len(self.all_x))
			elif isinstance(train_samples, float):
				self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.all_x, self.all_y, train_size = train_samples)

		else:

			if not test_samples:
				test_samples = 0.3

			if isinstance(test_samples, int):
				self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.all_x, self.all_y, test_size = test_samples / len(self.all_x))
			elif isinstance(test_samples, float):
				self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.all_x, self.all_y, test_size = test_samples)

		self.dimensions = {'input': {'train': self.train_x.shape, 'test': self.test_x.shape}, 
						   'output': {'train': self.train_y.shape, 'test': self.test_y.shape}}
