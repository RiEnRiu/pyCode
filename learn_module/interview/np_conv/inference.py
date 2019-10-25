import numpy as np
import struct
import matplotlib.pyplot as plt

class minist:
    def __init__(self):
        with open('data/train-images.idx3-ubyte', 'rb') as train_data_path:
            _, num, rows, cols = struct.unpack('>IIII',train_data_path.read(16))
            self.train_data = np.fromfile(train_data_path, dtype=np.uint8)
            self.train_data = np.reshape(self.train_data, (num, rows, cols))
            self.train_data = self.train_data.astype(np.float32)
            self.train_data = self.train_data / 255
        with open('data/train-labels.idx1-ubyte', 'rb') as train_label_path:
            _, _ = struct.unpack('>II', train_label_path.read(8))
            self.train_label = np.fromfile(train_label_path, dtype=np.uint8)
            self.train_label = self.train_label.astype(np.int)
        with open('data/t10k-images.idx3-ubyte', 'rb') as test_data_path:
            _, num, rows, cols = struct.unpack('>IIII', test_data_path.read(16))
            self.test_data = np.fromfile(test_data_path, dtype=np.uint8)
            self.test_data = np.reshape(self.test_data, (num, rows, cols))
            self.test_data = self.test_data.astype(np.float32)
            self.test_data = self.test_data / 255
        with open('data/t10k-labels.idx1-ubyte', 'rb') as test_label_path:
            _, _ = struct.unpack('>II', test_label_path.read(8))
            self.test_label = np.fromfile(test_label_path, dtype=np.uint8)
            self.test_label = self.test_label.astype(np.int)

    def get_train_set(self):
        return self.train_data, self.train_label

    def get_test_set(self):
        return self.test_data, self.test_label

    def show_train_image(self, index):
        img = self.train_data[index,:,:]
        plt.imshow(img)
        plt.show()

    def show_test_image(self, index):
        img = self.test_data[index,:,:]
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    xxx = minist()
    xxx.show_train_image(0)