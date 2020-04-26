import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
import os

path_inception = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = path_inception

pre_trained_model = InceptionV3(input_shape= (150,150,3),
                               include_top = False,
                               weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output

#Ending training once we reach 97% accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') > 0.97):
            print("\nReached 97% accuracy so cancelling training!")
            self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(10, activation = 'softmax')(x)

model = Model(pre_trained_model.input,x)

model.compile(optimizer = RMSprop(lr = 0.0001),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   batch_size = 20,
                                                   class_mode = 'categorical',
                                                   target_size = (150,150))

test_generator =  test_datagen.flow_from_directory(test_dir,
                                                        batch_size = 20,
                                                        class_mode = 'categorical',
                                                        target_size = (150,150))

callbacks = myCallback()
history = model.fit(train_generator,
                              validation_data = test_generator,
                              steps_per_epoch = 125,
                              epochs = 100,
                              verbose = 2,
                              validation_steps = 50,
                              callbacks = [callbacks])

from keras.models import load_model
model.save('/model/model.h5')
