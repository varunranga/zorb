#!/usr/local/bin/python3.8

import argparse
parser = argparse.ArgumentParser(description = 'Use ZORB to train a deep neural network using cli.')
parser.add_argument('-d', '--dataset', type = str, default = "MNIST", help = 'Dataset to train the network (see zorb.datasets)')
parser.add_argument('-n', '--network', nargs='*', help = 'Neural network architecture (only Sequential model)')
parser.add_argument('-rm', '--train-samples', type = int, help = 'Number of training samples from training set used in training (see specific dataset object)')
parser.add_argument('-em', '--test-samples', type = int, help = 'Number of testing samples from testing set used in evaluation (see specific dataset object)')
parser.add_argument('-s', '--save', type = str, help = 'Filename to save the network', default = False)
parser.add_argument('-l', '--load', type = str, help = 'Filename to load the network from', default = False)
parser.add_argument('-nt', '--no-train', action = 'store_true', help = "Do not train the model")
parser.add_argument('-ne', '--no-evaluate', action = 'store_true', help = "Do not evaluate the model")
args = parser.parse_args()

import zorb

from time import time

dataset = eval("zorb.datasets." + args.dataset)(train_samples = args.train_samples, test_samples = args.test_samples)

if args.load: network = zorb.models.load(args.load)

else:

	network = [layer.replace('[', '(').replace(']', ')') if ('[' in layer) and (']' in layer) else (layer + "()") for layer in args.network]	
	network = [eval("zorb.layers."+layer) for layer in network]
	
	model = zorb.models.Sequential(input_shape = dataset.dimensions['input']['train'][1:], network = network)

model.summary()

if not args.no_evaluate:

	print("EVALUATING")
	print("Train:", model.evaluate(X = dataset.train_x, Y = dataset.train_y))
	print("Test:", model.evaluate(X = dataset.test_x, Y = dataset.test_y))

if not args.no_train:

	print("TRAINING")

	st = time()

	model.fit(X = dataset.train_x, Y = dataset.train_y)

	et = time()

	print(f"Model trained in {et-st:.4f} seconds")

if not args.no_evaluate:

	print("EVALUATING")
	print("Train:", model.evaluate(X = dataset.train_x, Y = dataset.train_y))
	print("Test:", model.evaluate(X = dataset.test_x, Y = dataset.test_y))
