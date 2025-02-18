from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def AlexNet(input_shape, num_classes):
    alex = Sequential([
        Conv2D(96, 11, strides=(4,4), activation='relu', input_shape=input_shape),

        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(256, 5, padding='same', activation='relu'),

        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(256, 3, padding='same', activation='relu'),

        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name='AlexNet')
    return alex
