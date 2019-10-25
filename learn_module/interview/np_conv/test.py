import numpy as np
from fully_connecte import fully_connecte
from conv import conv
from loss import softmax_loss
from pooling import max_pooling
from activation import relu
from inference import minist
import time

data = minist()
test_set = data.get_test_set()
test_imgs, test_label = test_set

batch_size = 50


conv1 = conv((batch_size, 28, 28, 1), (1, 5, 5, 12), name='conv1')
relu1 = relu()
pool1 = max_pooling([batch_size, 24, 24, 12], (2, 2), stride=2)

conv2 = conv((batch_size, 12, 12, 12), (12, 5, 5, 20), name='conv2')
relu2 = relu()
pool2 = max_pooling([batch_size, 8, 8, 20], (2, 2), stride=2)

fc1 = fully_connecte(batch_size, 20*4*4, 100, name='fc1')
relu3 = relu()
fc2 = fully_connecte(batch_size, 100, 10, name='fc2')

conv1.restore('params/conv1')
conv2.restore('params/conv2')
fc1.restore('params/fc1')
fc2.restore('params/fc2')

print('please wait...')
start_time = time.clock()
correct = 0
test_num = int(test_imgs.shape[0] / batch_size)
for i in range(test_num):
    input_imgs = test_imgs[i * batch_size:(i + 1) * batch_size, :, :]
    input_imgs = input_imgs.reshape([batch_size, 28, 28, 1])
    input_y = test_label[i * batch_size:(i + 1) * batch_size]
    out = conv1.forward(input_imgs)
    out = relu1.forward(out)
    out = pool1.forward(out)

    out = conv2.forward(out)
    out = relu2.forward(out)
    out = pool2.forward(out)

    out = out.reshape([batch_size, -1]).T

    out = fc1.forward(out)
    out = relu3.forward(out)
    out = fc2.forward(out)

    correct += np.sum(out.argmax(axis=0) == input_y)
end_time = time.clock()
print('spend time: %f' % (end_time - start_time))
print('accuracy: %s' % (correct / 10000.0))