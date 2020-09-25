import networkf as network
import mnist_loader
import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image

# imgc = Image.open('num3.jpg')
# nimgc = imgc.resize((28,28))
# nimgc.save('num4.jpg')
# img.show()

# img_org = cv2.imread('num4.jpg')
# img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
# print(type(img_gray))
# print((np.shape(img_gray)))
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_l = [x for x in training_data]
test_data_l = [x for x in test_data]
# img = np.reshape(img_gray,(784,1))
net = network.Network([784, 100, 10])
print('Started')
net.SGD(training_data_l, 30, 20, 3.0, test_data=test_data_l)
np.save('trainedb.npy', net.biases, allow_pickle=True)
np.save('trainedw.npy', net.weights, allow_pickle=True)
print('Done')
# res = net.feedforward(img)
# print(np.argmax(res))
# print((np.shape(img)))
# plt.subplot(1,1,1)
# plt.imshow(img_gray)
# plt.show()



