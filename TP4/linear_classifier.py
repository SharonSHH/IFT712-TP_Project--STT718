import numpy as np


class LinearClassifier(object):
	def __init__(self, x_train, y_train, x_val, y_val, num_classes, bias=False):
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val
		self.bias = bias  # when bias is True then the feature vectors have an additional 1

		num_features = x_train.shape[1]
		if bias:
			num_features += 1

		self.num_features = num_features
		self.num_classes = num_classes
		self.W = self.generate_init_weights(0.01)

	def generate_init_weights(self, init_scale):
		return np.random.randn(self.num_features, self.num_classes) * init_scale

	def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, init_scale=0.01):
		"""
		Train the model with a cross-entropy loss
		Naive implementation (with loop)

		Inputs:
		- num_epochs: the number of training epochs
		- lr: learning rate
		- l2_reg: the l2 regularization strength
		- lr_decay: learning rate decay.  Typically a value between 0 and 1
		- init_scale : scale at which the parameters self.W will be randomly initialized

		Returns a tuple for:
		- training accuracy for each epoch
		- training loss for each epoch
		- validation accuracy for each epoch
		- validation loss for each epoch
		"""
		loss_train_curve = []
		loss_val_curve = []
		accu_train_curve = []
		accu_val_curve = []

		self.W = self.generate_init_weights(init_scale)  # type: np.ndarray

		sample_idx = 0
		num_iter = num_epochs * len(self.x_train)
		for i in range(num_iter):
			# Take a sample
			x_sample = self.x_train[sample_idx]
			y_sample = self.y_train[sample_idx]
			if self.bias:
				x_sample = augment(x_sample)

			# Compute loss and gradient of loss
			loss_train, dW = self.cross_entropy_loss(x_sample, y_sample, l2_reg)

			# Take gradient step
			self.W -= lr * dW

			# Advance in data
			sample_idx += 1
			if sample_idx >= len(self.x_train):  # End of epoch

				accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train, l2_reg)
				accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val, l2_reg)

				loss_train_curve.append(loss_train)
				loss_val_curve.append(loss_val)
				accu_train_curve.append(accu_train)
				accu_val_curve.append(accu_val)

				sample_idx = 0
				lr *= lr_decay

		return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

	def predict(self, X):
		"""
		return the class label with the highest class score i.e.

			argmax_c W.X

		 X: A numpy array of shape (D,) containing one or many samples.

		 Returns a class label for each sample (a number between 0 and num_classes-1)
		"""
		if self.bias:
			X = augment(X)
		output = np.dot(X,self.W)
		return np.argmax(output,axis=1)

	def global_accuracy_and_cross_entropy_loss(self, X, y, reg=0.0):
		"""
		Compute average accuracy and cross_entropy for a series of N data points.
		Naive implementation (with loop)
		Inputs:
		- X: A numpy array of shape (D, N) containing many samples.
		- y: A numpy array of shape (N) labels as an integer
		- reg: (float) regularization strength
		Returns a tuple of:
		- average accuracy as single float
		- average loss as single float
		"""
		accu = 0.0
		loss = 0.0
		N = y.shape[0]

		if self.bias and X.shape[1] < self.W.shape[0]:
			X = augment(X)

		outputs = np.dot(X,self.W)

		for n in range(N):
			output = outputs[n]
			#To avoid overflow
			threshold = np.vectorize(lambda t : max(-1000, min(t,1000)))
			output = threshold(output)
			#To avoid exponential overflow
			softmax = np.exp(output - np.max(output))
			softmax = softmax/np.sum(softmax)
			threshold = np.vectorize(lambda t : max(t,0.001))
			softmax = threshold(softmax)
			loss -= np.log(softmax[y[n]])
			loss += reg*np.linalg.norm(self.W)
			#Accuracy
			if np.argmax(output, axis=0) == y[n]:
				accu += 1

		loss /= N
		accu /= N

		return accu, loss

	def cross_entropy_loss(self, x, y, reg=0.0):
		"""
		Cross-entropy loss function for one sample pair (X,y) (with softmax)
		C.f. Eq.(4.104 to 4.109) of Bishop book.

		Input have dimension D, there are C classes.
		Inputs:
		- W: A numpy array of shape (D, C) containing weights.
		- x: A numpy array of shape (D,) containing one sample.
		- y: training label as an integer
		- reg: (float) regularization strength
		Returns a tuple of:
		- loss as single float
		- gradient with respect to weights W; an array of same shape as W
		"""

		#1
		output = np.dot(self.W.T,x)
		#To avoid overflow
		threshold = np.vectorize(lambda t : max(-1000, min(t,1000)))
		output = threshold(output)
		#To avoid exponential overflow
		softmax = np.exp(output - np.max(output))
		softmax = softmax/np.sum(softmax)
		threshold = np.vectorize(lambda t : max(t,0.001))
		softmax = threshold(softmax)
		#2 / 3
		#As the target is a one-hot vector, only the score of the true class matters
		loss = -np.log(softmax[y])
		loss += reg*np.linalg.norm(self.W)

		#4
		one_hot_y = np.zeros_like(output)
		one_hot_y[y] = 1
		dW = np.tile(x,(output.shape[0],1)).T
		dW = (output-one_hot_y)*dW
		dW += (reg/2)*self.W

		return loss, dW


def augment(x):
	if len(x.shape) == 1:
		return np.concatenate([x, [1.0]])
	else:
		return np.concatenate([x, np.ones((len(x), 1))], axis=1)
