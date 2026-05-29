"""Arquitetura 2: CNN Moderna (estilo VGG simplificado, com BatchNorm e Dropout).

Empilha blocos convolucionais com filtros 3x3, normalização em lote
(BatchNormalization) para acelerar e estabilizar o treino, e Dropout para
regularização. É uma arquitetura mais profunda e robusta que a LeNet.
"""
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, Dropout,
                          Flatten, Dense)

from common import treinar_e_avaliar, IMAGE_ROWS, IMAGE_COLS, CORES, NUM_CLASSES


def construir_modelo():
    input_shape = (IMAGE_ROWS, IMAGE_COLS, CORES)
    model = Sequential(name="CNN_Moderna")

    # Bloco 1
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Bloco 2
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classificador
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    return model


if __name__ == "__main__":
    treinar_e_avaliar(construir_modelo(), "CNN_Moderna", achatar=False)
