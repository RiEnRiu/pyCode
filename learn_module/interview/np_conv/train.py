import numpy as np
from fully_connecte import fully_connecte
from conv import conv
from loss import softmax_loss
from pooling import max_pooling
from activation import relu
from inference import minist
import time

data = minist()
train_set = data.get_train_set()
test_set = data.get_test_set()
train_imgs, train_label = train_set
test_imgs, test_label = test_set

batch_size = 50
learning_rate = 0.001


conv1 = conv((batch_size, 28, 28, 1), (1, 5, 5, 12), name='conv1')
relu1 = relu()
pool1 = max_pooling([batch_size, 24, 24, 12], (2, 2), stride=2)
conv2 = conv((batch_size, 12, 12, 12), (12, 5, 5, 20), name='conv2')
relu2 = relu()
pool2 = max_pooling([batch_size, 8, 8, 20], (2, 2), stride=2)
fc1 = fully_connecte(batch_size, 20*4*4, 100, name='fc1')
relu3 = relu()
fc2 = fully_connecte(batch_size, 100, 10, name='fc2')
softmaxloss = softmax_loss(batch_size, 10)

num = int(train_imgs.shape[0] / batch_size)
for i in range(200):
    for j in range(num):
        input_imgs = train_imgs[j * batch_size:(j + 1) * batch_size, :, :]
        input_imgs = input_imgs.reshape([batch_size, 28, 28, 1])
        label = train_label[j * batch_size:(j + 1) * batch_size]

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

        loss = softmaxloss.forward(out, label)
        df = softmaxloss.backward()

        df = fc2.backward(df, learning_rate)
        df = relu3.backward(df)
        df = fc1.backward(df, learning_rate)
        df = df.T.reshape([batch_size, 4, 4, 20])

        df = pool2.backward(df)
        df = relu2.backward(df)
        df = conv2.backward(df, learning_rate)

        df = pool1.backward(df)
        df = relu1.backward(df)
        df = conv1.backward(df, learning_rate)

        print('epoch:%d, step: %d, loss: %lf' % (i, j, loss))

    conv1.store('params/')
    conv2.store('params/')
    fc1.store('params/')
    fc2.store('params/')

    correct = 0
    test_num = int(test_imgs.shape[0] / batch_size)
    for i in range(test_num):  #
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
    print('accuracy: %s' % (correct / 10000.0))