import numpy as np

class max_pooling:
    def __init__(self, input_shape, kernel_size, stride = 2):
        self.batch_size, self.in_w, self.in_h, self.in_c = input_shape
        self.k_w, self.k_h = kernel_size
        self.stride = stride
        self.x = np.zeros(input_shape)

    def forward(self, input):
        self.com_2d = np.zeros([self.batch_size * self.in_c * int((self.in_w / self.k_w)) * int((self.in_h / self.k_h)), self.k_w * self.k_h])
        index = 0
        for n in range(self.batch_size):
            for i in range(int(self.in_w / self.k_w)):
                for j in range(int(self.in_h / self.k_h)):
                    x_block = input[n, i * self.k_w:(i + 1) * self.k_w, j * self.k_h:(j + 1) * self.k_h,:]
                    self.com_2d[index * self.in_c:(index + 1) * self.in_c, :] = x_block.reshape(self.k_w * self.k_h, self.in_c).T
                    index += 1
        self.pooling_X = np.max(self.com_2d, axis=1).reshape([-1, 1])
        self.max_index = np.argmax(self.com_2d, axis=1)

        self.pooling_images = np.reshape(self.pooling_X,[self.batch_size, int(self.in_w / self.k_w), int(self.in_h / self.k_h), self.in_c])
        return self.pooling_images

    def backward(self, df):
        df_reshape = df.reshape([-1])

        d_com_2d = np.zeros(self.com_2d.shape)
        d_com_2d[range(self.com_2d.shape[0]), self.max_index] = df_reshape
        dx = np.zeros(self.x.shape)
        index = 0
        for n in range(self.batch_size):
            for i in range(int(self.in_w / self.k_w)):
                for j in range(int(self.in_h / self.k_h)):
                    d_com_2d_block = d_com_2d[index * self.in_c:(index + 1) * self.in_c,:]
                    d_com_2d_block = d_com_2d_block.T.reshape(self.k_w, self.k_h, self.in_c)
                    dx[n, i * self.k_w:(i + 1) * self.k_w, j * self.k_h:(j + 1) * self.k_h, :] = d_com_2d_block
                    index += 1
        return dx