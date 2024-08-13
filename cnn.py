import numpy
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

a1 = X_test[1]
a2 = a1.reshape(1, 1, 28, 28)
model_pred = model.predict_classes(a2, verbose=0)
print('Prediction of loaded_model: {}'.format(model_pred[0]))

import matplotlib.pyplot as plt

test_images = X_test[1:4]
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print("test images shape - {}".format(test_images.shape))

for i, test_image in enumerate(test_images, start=1):
	org_image = test_image
	test_image = test_image.reshape(1, 1, 28, 28)
	prediction = model.predict_classes(test_image, verbose=0)
	print("I think the digit is - {}".format(prediction[0]))
	plt.subplot(220+i)
	plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
image_test = cv2.imread('images/5a.jpg', 0)
image_test_1 = cv2.resize(image_test, (28, 28))
image_test_2 = image_test_1.reshape(1, 1, 28, 28)

cv2.imshow('number', image_test_1)
loaded_model_pred = loaded_model.predict_classes(image_test_2, verbose=0)
print('Prediction of loaded_model: {}'.format(loaded_model_pred[0]))