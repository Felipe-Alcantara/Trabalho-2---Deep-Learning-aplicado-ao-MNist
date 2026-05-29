"""Arquitetura 3: MLP (Perceptron Multicamadas / rede totalmente conectada).

Diferente das CNNs, o MLP ignora a estrutura espacial da imagem: cada um dos
784 pixels (28x28) vira uma entrada de um vetor. Serve como linha de base para
mostrar o ganho que as convoluções trazem em problemas de visão.
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout

from common import treinar_e_avaliar, IMAGE_ROWS, IMAGE_COLS, NUM_CLASSES


def construir_modelo():
    entrada = IMAGE_ROWS * IMAGE_COLS  # 784
    model = Sequential(name="MLP")
    model.add(Dense(512, activation="relu", input_shape=(entrada,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    return model


if __name__ == "__main__":
    treinar_e_avaliar(construir_modelo(), "MLP", achatar=True)
