import numpy as np

class Perceptron:

	def __init__(self, inputs, outputs, learning_rate=0.1, epochs=100):
		self.inputs = inputs
		self.outputs = outputs
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = np.random.rand(len(self.inputs[0]), len(self.outputs[0]))
		self.bias = np.random.rand(len(self.outputs[0]))
		
	def train(self):
		for h in range(self.epochs):
			for i in range(len(self.inputs)):
				delta = self.getDelta(i)
				for j in range(len(self.outputs[0])):
					self.updateWeightsBiases(j, self.inputs[i], delta[j])

	def getDelta(self, i):
		y = self.getY(self.inputs[i])
		return np.subtract(self.outputs[i], y)
	
	def getY(self, input):
		v = np.add(np.matmul(input, self.weights), self.bias)
		v[v>=0] = 1
		v[v<0] = 0
		return v
	
	def updateWeightsBiases(self, j, input, delta):
		self.weights[:,j] += self.learning_rate * delta * input
		self.bias[j] += self.learning_rate * delta