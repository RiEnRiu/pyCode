import numpy as np


class fully_connecte:
    def __init__(self, batch_size, in_num, out_num, name):
        # property init
        self.batch_size = batch_size
        self.in_num = in_num
        self.out_num = out_num
        self.name = name
        # params init
        self.w = np.random.normal(0, 1, (self.out_num, self.in_num)) * 0.01
        self.b = np.zeros((self.out_num, 1))
        # cache init
        self.x = np.zeros((self.in_num, batch_size))

    def forward(self, input):
        self.x[...] = input[...]
        return np.dot(self.w, self.x) + self.b

    def backward(self, df, lr):
        dx = np.dot(self.w.T, df)
        dw = np.dot(df, self.x.T)
        db = np.sum(df, axis=1).reshape([self.out_num, 1])
        self.w = self.w - lr * dw
        self.b = self.b - lr * db
        return dx

    def store(self, dir_path):
        str_w = dir_path + self.name + '_w.npy'
        str_b = dir_path + self.name + '_b.npy'
        np.save(str_w, self.w)
        np.save(str_b, self.b)

    def restore(self, file_pre_name):
        str_w = file_pre_name + '_w.npy'
        str_b = file_pre_name + '_b.npy'
        self.w = np.load(str_w)
        self.b = np.load(str_b)