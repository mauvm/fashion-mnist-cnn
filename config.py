import numpy as np
from tensorflow import keras

# See Context paragraph on https://www.kaggle.com/zalando-research/fashionmnist
labels = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot',
]
num_classes = len(labels)

epochs = 20
batch_size = 200
image_width = 28
image_height = 28

train_file = '../large_files/fashionmnist/fashion-mnist_train.csv'
test_file = '../large_files/fashionmnist/fashion-mnist_test.csv'

model_file = 'my_model.h5'

def preprocessDataFrame(df):
  data = df.values
  X = data[:, 1:] # All pixel{N} columns
  Y = data[:, 0] # First column

  # Reshape (N, 784) => (N, 28, 28, 1) because Conv2D layers require 4 dimensions
  X = np.reshape(X, (len(X), image_width, image_height, 1))

  return X, Y

def oneHotEncodeY(Y):
  return keras.utils.to_categorical(Y, num_classes=num_classes)