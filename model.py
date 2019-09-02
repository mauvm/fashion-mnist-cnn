from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import config

def createModel():
  model = Sequential()
  model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu',
                   input_shape = (config.image_width, config.image_height, 1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(config.num_classes, activation='softmax'))

  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

  return model