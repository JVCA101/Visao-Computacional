from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Sequential, Module, Dropout, BatchNorm2d

class FrankNet(Module):
  def __init__(self):
    super(FrankNet, self).__init__()
    self.conv_layer = Sequential(

      Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      ReLU(),
      Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=2, stride=2),
      Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      ReLU(),
      Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=2, stride=2),
      Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      ReLU(),
      Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      ReLU(),
      Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=2, stride=2),


      Conv2d(256, 384, kernel_size=3, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=3, stride=2),
      Conv2d(384, 384, kernel_size=3, padding=1),
      ReLU(),
      Conv2d(384, 256, kernel_size=3, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=3, stride=2)
    )
    
    self.flatten = Flatten()
    
    self.full_layer = Sequential(
      Linear(6*6*256, 4096),
      ReLU(),
      Linear(4096, 2048),
      ReLU(),
      Linear(2048, 10)
    )
    
  def forward(self, x):
    out = self.conv_layer(x)
    # out = out.reshape(out.size(0), -1)
    out = self.flatten(out)
    out = self.full_layer(out)
    return out


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# def FrankNet(input_shape, num_classes):
#     frank = Sequential([
#         # Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=input_shape),
#         # Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),

#         # MaxPooling2D(pool_size=2, strides=2),
#         # Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),
#         # Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),

#         # MaxPooling2D(pool_size=2, strides=2),
#         # Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),
#         # Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),
#         # Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu'),

#         Conv2D(filters=96, kernel_size=11, strides=(4,4), activation='relu', input_shape=input_shape),

#         MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#         Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),

#         MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#         Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
#         Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
#         Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),



#         MaxPooling2D(pool_size=2, strides=2),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),

#         MaxPooling2D(pool_size=2, strides=2),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),
#         Conv2D(filters=512, kernel_size=3, padding='same', strides=1, activation='relu'),

#         MaxPooling2D(pool_size=2, strides=2),

#         Flatten(),
#         Dense(4096, activation='relu'),
#         Dense(4096, activation='relu'),
#         Dense(num_classes, activation='softmax')
#     ], name='FrankNet')
#     return frank
