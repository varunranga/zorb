from numpy import sqrt
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import hstack
from numpy import vstack
from numpy import zeros
from numpy import ones
from numpy import reshape
from numpy.random import rand
from sklearn.model_selection import train_test_split

class TwoSpirals():

	def __init__(self, train_samples = 150, test_samples = 50, noise = 0.0):

		n_points = train_samples

		n = sqrt(rand(n_points, 1)) * 780 * (2 * pi) / 360
		d1x = -cos(n) * n + rand(n_points, 1) * noise
		d1y = sin(n) * n + rand(n_points, 1) * noise
		X, Y = (vstack((hstack((d1x, d1y)), hstack((-d1x, -d1y)))), 
				hstack((zeros(n_points), ones(n_points))))

		Y = reshape(Y, (-1, 1))

		self.train_x = X
		self.train_y = Y

		n_points = test_samples

		n = sqrt(rand(n_points, 1)) * 780 * (2 * pi) / 360
		d1x = -cos(n) * n + rand(n_points, 1) * noise
		d1y = sin(n) * n + rand(n_points, 1) * noise
		X, Y = (vstack((hstack((d1x, d1y)), hstack((-d1x, -d1y)))), 
				hstack((zeros(n_points), ones(n_points))))

		Y = reshape(Y, (-1, 1))

		self.test_x = X
		self.test_y = Y

		self.dimensions = {'input': {'train': self.train_x.shape, 'test': self.test_x.shape}, 
						   'output': {'train': self.train_y.shape, 'test': self.test_y.shape}}
