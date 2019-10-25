import numpy as np

class conv:
    def __init__(self, input_shape, kernel_shape, stride=1, name=None):
        # property init
        assert (name != None)
        self.batch_size, self.in_w, self.in_h, self.in_c = input_shape
        _, self.f_w, self.f_h, self.out_c = kernel_shape
        self.stride = stride
        self.name = name
        self.out_h = int((self.in_h - self.f_h) / self.stride + 1)
        self.out_w = int((self.in_w - self.f_w) / self.stride + 1)
        self.k_w = self.f_w * self.f_h * self.in_c
        # weights init
        low = -1.0 * np.sqrt(6.0 / (self.k_w + self.out_c))
        high = 1.0 * np.sqrt(6.0 / (self.k_w + self.out_c))
        self.w_2d = np.random.uniform(low=low, high=high, size=(self.k_w, self.out_c))
        self.b = np.zeros((self.out_c,))
        # cache init
        self.com_2d = np.zeros((self.batch_size * self.out_h * self.out_w, self.k_w))
        self.x = np.zeros(input_shape)

    def forward(self, input):
        self.x[...] = input[...]
        index = 0
        for num in range(self.batch_size):
            for i in range(self.in_w - self.f_w + 1):
                for j in range(self.in_h - self.f_h + 1):
                    self.com_2d[index, :] = self.x[num, i:i + self.f_w, j:j + self.f_h, :].reshape(self.k_w)[:]
                    index += 1
        out = np.dot(self.com_2d, self.w_2d)[...]
        out = out.reshape((self.batch_size, self.out_w, self.out_h, self.out_c))
        out = out + self.b[np.newaxis, np.newaxis, np.newaxis, :]
        return out

    def backward(self, df, lr):
        # df: (batch_size, out_w, out_h, out_c)
        # reshape
        df_reshape = df.reshape([self.batch_size * self.out_h * self.out_w, self.out_c])
        d_com_2d = np.dot(df_reshape, self.w_2d.T)
        dw_2d = np.dot(self.com_2d.T, df_reshape)
        db = np.sum(df, axis=(0, 1, 2))

        dx = np.zeros((self.batch_size, self.in_w, self.in_h, self.in_c))
        index = 0
        for num in range(self.batch_size):
            for i in range(self.in_w - self.f_w + 1):
                for j in range(self.in_h - self.f_h + 1):
                    dx[num, i:i + self.f_w, j:j + self.f_h, :] += d_com_2d[index, :].reshape( self.f_w, self.f_h, self.in_c)
                    index += 1

        self.w_2d = self.w_2d - lr * dw_2d
        self.b = self.b - lr * db
        return dx

    def store(self, dir_path):
        str_w = dir_path + self.name + '_w.npy'
        str_b = dir_path + self.name + '_b.npy'
        np.save(str_w, self.w_2d)
        np.save(str_b, self.b)

    def restore(self, file_pre_name):
        str_w = file_pre_name + '_w.npy'
        str_b = file_pre_name + '_b.npy'
        self.w_2d = np.load(str_w)
        self.b = np.load(str_b)
        