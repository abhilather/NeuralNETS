# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:36:22 2018

@author: Abhimanyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:56:59 2018

@author: Abhimanyu
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

#classifier.add(Convolution2D(32, 3, 3, activation='relu'))

#classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 256, activation='relu'))
Dropout(rate=0.4)


classifier.add(Dense(output_dim = 74, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



from keras.preprocessing.image import ImageDataGenerator

training_set = ImageDataGenerator(
        rescale=1./255)

test_set = ImageDataGenerator(rescale=1./255)

training_set= training_set.flow_from_directory(
        'Training',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical')

test_set = test_set.flow_from_directory(
        'Test',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=37101,
        epochs=5,
        validation_data=test_set,
        validation_steps=12460)