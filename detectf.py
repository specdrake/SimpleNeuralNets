import networkf as network
import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# imgc = Image.open('num2.jpg')
# nimgc = imgc.resize((28,28))
# nimgc.save('numtest.jpg')
# img.show()

img_org = cv2.imread('numtest.jpg')
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
# print(type(img_gray))
# print((np.shape(img_gray)))
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_l = [x for x in training_data]
test_data_l = [x for x in test_data]
img = np.reshape(img_gray,(784,1))
net = network.Network([784, 15, 10])
w = np.load('trainedw.npy', allow_pickle=True)
b = np.load('trainedb.npy', allow_pickle=True)
net.biases = b
net.weights = w
# net.SGD(training_data_l, 30, 10, 10.0, test_data=test_data_l)
res = net.feedforward(img)
print(res)
print(np.argmax(res))
# print((np.shape(img)))
# plt.subplot(1,1,1)
# plt.imshow(img_gray)
# plt.show()



