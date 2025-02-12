import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def VGG13(input_shape, num_classes):
    vgg13 = Sequential([
        Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=input_shape),
        Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),

        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),

        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),

        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),

        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),

        MaxPooling2D(pool_size=2, strides=2),

        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name='VGG13')
    return vgg13
