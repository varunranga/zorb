import numpy

class XOR():

	def __init__(self, train_samples = None, test_samples = None):


		X = []
		Y = []
		for line in open('./datasets/XOR/xor_train.txt').readlines():
			line = line.strip()
			y, x = line.split('\t')
			x0, x1 = x.split(' ')
			x0, x1, y = float(x0), float(x1), float(y)
			X.append([x0, x1])
			Y.append([y])

		self.train_x = numpy.array(X)
		self.train_y = numpy.array(Y)

		X = []
		Y = []
		for line in open('./datasets/XOR/xor_test.txt').readlines():
			line = line.strip()
			y, x = line.split('\t')
			x0, x1 = x.split(' ')
			x0, x1, y = float(x0), float(x1), float(y)
			X.append([x0, x1])
			Y.append([y])

		self.test_x = numpy.array(X)
		self.test_y = numpy.array(Y)

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
