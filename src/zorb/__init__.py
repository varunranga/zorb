from enum import Enum

class enum(Enum):

	Dense = 1
	Parametric = 2
	Sigmoid = 3
	Activation = 4
	TrainingPhase = 5
	Flatten = 6
	NonParametric = 7
	TestingPhase = 8
	Convolution2D = 9
	Tanh = 10
	ReLU = 11
	Softmax = 12
	ForwardNormalize = 13

del Enum

enum_to_layer = {key.value: key.name for key in enum}

from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
from jax import random

seed = np.random.randint(np.iinfo(np.int64).max)
key = random.PRNGKey(seed)

from . import models
from . import datasets
from . import layers
