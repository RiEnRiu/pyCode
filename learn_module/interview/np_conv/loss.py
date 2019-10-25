import numpy as np

class softmax_loss:
    def __init__(self, batch_size, in_num):
        self.batch_size = batch_size
        self.in_num = in_num
        self.loss = 0.0
        self.label = None
        self.out = np.zeros((in_num, batch_size))
        self.df = np.zeros((in_num, batch_size))

    def forward(self, input, label):
        self.out = input[...]
        self.label = label
        self.out = self.out - np.max(self.out, axis=0).reshape([1, self.batch_size])
        self.out = np.exp(self.out)
        self.out /= np.sum(self.out, axis=0).reshape([1, self.batch_size])

        self.loss = 0.0
        for i in range(self.batch_size):
            self.loss += np.log(self.out[self.label[i], i])
        self.loss = -self.loss / self.batch_size
        return self.loss

    def backward(self):
        self.df[...] = self.out[...]
        for i in range(self.batch_size):
            self.df[self.label[i], i] -= 1
        return self.df