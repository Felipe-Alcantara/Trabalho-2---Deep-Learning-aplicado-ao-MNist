"""Arquitetura 1: LeNet (baseada no exemplo marceloarantes19/deepLearning).

LeNet é uma das primeiras CNNs (LeCun, 1998). Aqui replicamos a versão do
exemplo: dois blocos Conv+ReLU+MaxPooling seguidos de uma camada densa de 500
neurônios e a camada de saída softmax com 10 classes.
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

from common import treinar_e_avaliar, IMAGE_ROWS, IMAGE_COLS, CORES, NUM_CLASSES


def construir_modelo():
    input_shape = (IMAGE_ROWS, IMAGE_COLS, CORES)
    model = Sequential(name="LeNet")
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    treinar_e_avaliar(construir_modelo(), "LeNet", achatar=False)
