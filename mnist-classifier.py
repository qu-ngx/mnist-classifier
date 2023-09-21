import tensorflow as tf

# Import Tensorflow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_ds, test_ds = dataset['train'], dataset['test']

# Creating labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Getting number of train and test data set
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Preprocess the data 
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Part for getting 1 image from the dataset

# # Map function to map 1 return each
train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

for image, label in test_ds.take(1):
    break
image = image.numpy().reshape((28, 28))

# # Plot the image 
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 3 Layers in the network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
# Repeat forever throughout the dataset and shuffle
train_ds = train_ds.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

# Fitting a model 
model.fit(train_ds, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_ds, steps=math.ceil(num_test_examples/32))
print('Accuracy on test ds: ', test_acc)

# Make prediction from the trained model
for test_images, test_labels in test_ds.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print(predictions.shape)
print(predictions[0])
print(np.argmax(predictions[0]))