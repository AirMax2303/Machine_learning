import tensorflow as tf
from tensorflow import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка данных MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Нормализация значений пикселей к диапазону от 0 до 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание модели
model = models.Sequential()

# Входной слой
model.add(layers.Flatten(input_shape=(28, 28)))  # MNIST images are 28x28 pixels

# Скрытые полносвязные слои
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))

# Выходной слой
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for digits 0-9

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Размер изображений: 28x28
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Нормализация значений пикселей к диапазону от 0 до 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Разделение на тренировочный и тестовый наборы
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Создание модели
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Оценка производительности модели на тестовом наборе
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')