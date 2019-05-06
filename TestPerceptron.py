from PerceptronClass import Perceptron

import numpy as np

inputs = np.array([
	[0, 0, 0],
	[0, 0, 1],
	[0, 1, 0],
	[0, 1, 1],
	[1, 0, 0],
	[1, 0, 1],
	[1, 1, 0],
	[1, 1, 1],
])
outputs = np.array([
	[0, 1],
	[0, 1],
	[0, 1],
	[0, 1],
	[1, 0],
	[1, 0],
	[1, 0],
	[1, 0],
])

test = Perceptron(inputs, outputs, 0.1, 100)
test.train()
print(test.getY([0, 0, 0]))
print(test.getY([1, 0, 0]))
print(test.getY([0, 1, 1]))
print(test.getY([1, 1, 1]))