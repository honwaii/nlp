#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 0008 9:00
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : leeson06.py


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

tf.executing_eagerly()


def basic_operation():
    x = [[2.]]
    m = tf.matmul(x, x)
    print("x matmul x = {}".format(m))
    a = tf.constant([[1, 2],
                     [3, 4]])
    print(a)
    b = tf.add(a, 1)
    print(b)
    # element-wise multiplication
    print(a * b)
    print(tf.matmul(a, b))
    c = np.multiply(a, b)
    print(c)
    # Transfer a tensor to numpy array
    print(a.numpy())


def gradient():
    w = tf.Variable([[1.0]])
    with tf.GradientTape() as tape:
        loss = w * w
    grad = tape.gradient(loss, w)
    print(grad)
    return grad


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:10000, :, :]
    y_train = y_train[:10000]
    x_test = x_test[:1000, :, :]
    y_test = y_test[:1000]

    x_train = tf.cast(x_train[..., tf.newaxis] / 255, tf.float32),
    x_test = tf.cast(x_test[..., tf.newaxis] / 255, tf.float32),

    # y_train = y_train.astype('float32')
    # y_test = y_test.astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# Build the model using Sequential
def build_model1():
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    mnist_model.summary()
    return mnist_model


# Build the model using Model
def build_model2():
    inputs = tf.keras.Input(shape=(None, None, 1), name="digits")
    conv_1 = tf.keras.layers.Conv2D(16, [3, 3], activation="relu")(inputs)
    conv_2 = tf.keras.layers.Conv2D(16, [3, 3], activation="relu")(conv_1)
    ave_pool = tf.keras.layers.GlobalAveragePooling2D()(conv_2)
    outputs = tf.keras.layers.Dense(10)(ave_pool)
    mnist_model_2 = tf.keras.Model(inputs=inputs, outputs=outputs)
    mnist_model_2.summary()
    return mnist_model_2


def train(x_train, y_train, model):
    model = build_model1()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, shuffle=True, batch_size=128, epochs=3)
    return model


def predict(test, model):
    result = model.predict(np.asarray(test).reshape(1, 28, 28, 1))
    return result


# Use TF 2.0


def load_tf_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:10000, :, :]
    y_train = y_train[:10000]
    x_test = x_test[:1000, :, :]
    y_test = y_test[:1000]
    x_test = tf.cast(x_test[..., tf.newaxis] / 255, tf.float32)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def handle_dataset(x_train, y_train):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32),
         tf.cast(y_train, tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)
    return dataset


def tf_training(x_train, y_train):
    dataset = handle_dataset(x_train, y_train)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_history = []
    mnist_model = build_model1()
    for epoch in range(5):
        for (batch, (images, labels)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = mnist_model(images, training=True)
                loss_value = loss(labels, logits)
            grads = tape.gradient(loss_value, mnist_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
        print("Epoch {} finishted".format(epoch))
    return mnist_model


# Use keras
(x_train, y_train), (x_test, y_test) = load_dataset()
model = build_model1()
trained_model = train(x_train, y_train, model)
result = predict(x_test[0], trained_model)
print(result)

# Use TF 2.0
(x_train, y_train), (x_test, y_test) = load_tf_dataset()
model = tf_training(x_train, y_train)
result = predict(x_test[0], model)
print(y_test[0])
temp = result[0].tolist()
print(temp.index(max(temp)))
