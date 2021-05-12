from numpy import array
from numpy import arange
from numpy import sin
from numpy import reshape

class Sinc():

	def __init__(self, train_samples = (-10, 10), test_samples = (-30, 30), step = 0.1):

		if isinstance(train_samples, int) or isinstance(train_samples, float): (-train_samples, train_samples)

		X = array(arange(-train_samples[0], 0, step).tolist() + [0.0] + arange(step,  train_samples[1] + step, step).tolist())
		Y = array((sin(X[X < 0]) / X[X < 0]).tolist() + [1.0] + (sin(X[X > 0]) / X[X > 0]).tolist())

		X = reshape(X, (-1, 1))
		Y = reshape(Y, (-1, 1))

		self.train_x = X
		self.train_y = Y

		X = array(arange(-test_samples[0], 0, step).tolist() + [0.0] + arange(step,  test_samples[1] + step, step).tolist())
		Y = array((sin(X[X < 0]) / X[X < 0]).tolist() + [1.0] + (sin(X[X > 0]) / X[X > 0]).tolist())

		X = reshape(X, (-1, 1))
		Y = reshape(Y, (-1, 1))
		
		self.test_x = X
		self.test_y = Y

		self.dimensions = {'input': {'train': self.train_x.shape, 'test': self.test_x.shape}, 
							'output': {'train': self.train_y.shape, 'test': self.test_y.shape}}