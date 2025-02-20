from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def AlexNet(input_shape, num_classes):
    alex = Sequential([
        Conv2D(filters=96, kernel_size=11, strides=(4,4), activation='relu', input_shape=input_shape),

        MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),
        # Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),

        # MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        # Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        # Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        # Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),

        # MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
        # Flatten(),
        # Dense(4096, activation='relu'),
        # Dense(4096, activation='relu'),
        # Dense(num_classes, activation='softmax')

        # Layer 2: Convolutional Layer with ReLU and Max Pooling
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        # Layer 3: Convolutional Layer with ReLU
        Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Layer 4: Convolutional Layer with ReLU
        Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Layer 5: Convolutional Layer with ReLU and Max Pooling
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        # Flatten the output for fully connected layers
        Flatten(),

        # Fully Connected Layer 1
        Dense(4096, activation='relu'),
        Dropout(0.5),

        # Fully Connected Layer 2
        Dense(4096, activation='relu'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ], name='AlexNet')
    return alex
