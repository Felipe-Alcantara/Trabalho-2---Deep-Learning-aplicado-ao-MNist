"""Funções utilitárias compartilhadas pelas três soluções (LeNet, CNN moderna e MLP).

Carrega o MNIST, treina um modelo Keras por 10 épocas e salva os artefatos
(histórico de treino, matriz de confusão e acurácia) para posterior comparação.
"""
import json
import os

import numpy as np
import tensorflow as tf
from keras import utils as utls
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report

# Reprodutibilidade
tf.random.set_seed(42)
np.random.seed(42)

IMAGE_ROWS, IMAGE_COLS = 28, 28
NUM_CLASSES = 10
BATCH_SIZE = 256
EPOCHS = 10
CORES = 1  # canais de cor (MNIST é escala de cinza)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados")


def carregar_dados(achatar=False):
    """Carrega e normaliza o MNIST.

    achatar=True devolve vetores 784 (para o MLP);
    achatar=False devolve tensores 28x28x1 (para as CNNs).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if achatar:
        x_train = x_train.reshape(-1, IMAGE_ROWS * IMAGE_COLS)
        x_test = x_test.reshape(-1, IMAGE_ROWS * IMAGE_COLS)
    else:
        x_train = x_train.reshape(-1, IMAGE_ROWS, IMAGE_COLS, CORES)
        x_test = x_test.reshape(-1, IMAGE_ROWS, IMAGE_COLS, CORES)

    y_train_cat = utls.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = utls.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train_cat), (x_test, y_test_cat), y_test


def treinar_e_avaliar(model, nome, achatar=False):
    """Compila, treina por 10 épocas, avalia e salva os artefatos do modelo."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test), y_test_labels = carregar_dados(achatar)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    historico = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(x_test, y_test), verbose=2,
    )

    # Predições e matriz de confusão
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test_labels, y_pred)
    acc = float((y_pred == y_test_labels).mean())
    n_params = int(model.count_params())

    print(f"\n[{nome}] Acurácia no teste: {acc:.4f} | Parâmetros: {n_params:,}")
    print(classification_report(y_test_labels, y_pred, digits=4))

    artefatos = {
        "nome": nome,
        "acuracia_teste": acc,
        "parametros": n_params,
        "history": {k: [float(v) for v in vals] for k, vals in historico.history.items()},
        "matriz_confusao": cm.tolist(),
    }
    with open(os.path.join(RESULTS_DIR, f"{nome}.json"), "w") as f:
        json.dump(artefatos, f, indent=2)

    return artefatos
