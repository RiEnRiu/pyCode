import numpy as np

class relu:
    def __init__(self):
        self.x = None
        pass

    def forward(self, input):
        self.x = input.copy()
        return np.maximum(0, input)

    def backward(self, df):
        dx = df
        dx[self.x < 0] = 0
        return dx
    