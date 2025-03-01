from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, ReLU, Sequential, Module, Dropout, BatchNorm2d

class AlexNet(Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.conv_layer = Sequential(

      Conv2d(3, 96, kernel_size=11, stride=4),
      ReLU(),

      MaxPool2d(kernel_size=3, stride=2),
      Conv2d(96, 256, kernel_size=5, padding=2),
      ReLU(),

      MaxPool2d(kernel_size=3, stride=2),
      Conv2d(256, 384, kernel_size=3, padding=1),
      ReLU(),

      Conv2d(384, 384, kernel_size=3, padding=1),
      ReLU(),
      Conv2d(384, 256, kernel_size=3, padding=1),
      ReLU(),

      MaxPool2d(kernel_size=3, stride=2)
    )
    
    self.flatten = Flatten()
    
    self.full_layer = Sequential(
      Linear(9216, 4096),
      ReLU(),
      Linear(4096, 4096),
      ReLU(),
      Linear(4096, 10)
    )
    
  def forward(self, x):
    out = self.conv_layer(x)
    # out = out.reshape(out.size(0), -1)
    out = self.flatten(out)
    out = self.full_layer(out)
    return out
  
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# def AlexNet(input_shape, num_classes):
#     alex = Sequential([
#         Conv2D(filters=96, kernel_size=11, strides=(4,4), activation='relu', input_shape=input_shape),

#         MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),
#         Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),

#         MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#         Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
#         Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
#         Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),

#         MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
#         Flatten(),
#         Dense(4096, activation='relu'),
#         Dense(4096, activation='relu'),
#         Dense(num_classes, activation='softmax')

#     ], name='AlexNet')
#     return alex