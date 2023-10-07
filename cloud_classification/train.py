import tensorflow as tf

from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

batch_size = 16
img_size = 64
epochs=15
training_folder = 'clouds-train'
testing_folder = 'clouds-test'

# Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
  training_folder,
  color_mode='rgb',
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  testing_folder,
  color_mode='rgb',
  image_size=(img_size, img_size))

class_names = train_ds.class_names
print(class_names)

# Augment the training data to introduce more variety
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomZoom(0.1),
  layers.experimental.preprocessing.RandomContrast(0.1)
])

augmented_train_ds1 = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
augmented_train_ds2 = augmented_train_ds1.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.concatenate(augmented_train_ds1).concatenate(augmented_train_ds2)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Create the model
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the raw model to a .tflite file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model_raw.tflite', 'wb') as f:
  f.write(tflite_model)

# Save the quantized model to a .tflite file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open('model_quantized.tflite', 'wb') as f:
  f.write(quantized_tflite_model)

# Plot the training and validation accuracy at each epoch
epochs_range = range(epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()