import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


def create_model(learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, features, label, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    learning_rate = 0.003
    epochs = 50
    batch_size = 4000
    validation_split = 0.2

    model = create_model(learning_rate)
    epochs, hist = train_model(
        model, x_train, y_train, epochs, batch_size, validation_split)

    list_of_metrics = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics)

    print(x_test.shape)
    print(y_test.shape)

    print("\n Evaluate the model against the test set")
    model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
