from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, \
                                    Dense, Dropout, BatchNormalization

import config

def createModel():
  model = Sequential()
  
  model.add(Conv2D(64, (3, 3), strides=1, padding='same',
                   input_shape = (config.image_width, config.image_height, 1)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(256, (3, 3), strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())

  model.add(Dense(units=512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(units=128))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(units=config.num_classes))
  model.add(Activation('softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model