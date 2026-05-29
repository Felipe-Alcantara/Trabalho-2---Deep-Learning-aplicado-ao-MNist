"""Gera os quatro notebooks .ipynb da entrega a partir de um template comum.

Mantém os notebooks consistentes com os scripts de treino (mesma arquitetura,
mesmos hiperparâmetros) e no mesmo estilo do exemplo LeNet original.

Uso:
    python gerar_notebooks.py
"""
import json
import os

DIR = os.path.dirname(__file__)
RAIZ = os.path.dirname(DIR)

# Caminho do repositório no GitHub (usuário/repo). Ajuste após criar o repo da equipe.
REPO_GITHUB = "SUA-EQUIPE/Trabalho-2---Deep-Learning-aplicado-ao-MNist"


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": texto.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_COLAB = """\
# Setup — funciona no Google Colab (recomendado, com GPU) ou local.
# No Colab: Ambiente de execução > Alterar tipo > Acelerador de hardware: GPU.
import importlib, subprocess, sys

for pacote in ['seaborn', 'tensorflow']:
    if importlib.util.find_spec(pacote) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pacote])

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow', tf.__version__, '| GPU disponível:', bool(gpus), gpus or '(rodando em CPU)')"""

IMPORTS = """\
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import utils as utls
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

tf.random.set_seed(42)
np.random.seed(42)"""

HIPER = """\
# Hiperparâmetros (idênticos ao exemplo LeNet original)
imageRows, imageCols = 28, 28
numClasses = 10
batchSize = 256
epochs = 10
cores = 1  # canais de cor (MNIST é escala de cinza)"""


def bloco_dados(achatar):
    reshape = ("XTrain = XTrain.reshape(-1, imageRows * imageCols)\n"
               "XTest = XTest.reshape(-1, imageRows * imageCols)") if achatar else \
              ("XTrain = XTrain.reshape(-1, imageRows, imageCols, cores)\n"
               "XTest = XTest.reshape(-1, imageRows, imageCols, cores)")
    return f"""\
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
XTrain = XTrain.astype('float32') / 255.0
XTest = XTest.astype('float32') / 255.0
{reshape}

yTestLabels = yTest.copy()  # rótulos inteiros, usados na matriz de confusão
yTrain = utls.to_categorical(yTrain, numClasses)
yTest = utls.to_categorical(yTest, numClasses)
XTrain.shape, XTest.shape"""


TREINO = """\
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
historico = model.fit(XTrain, yTrain, batch_size=batchSize, epochs=epochs,
                      validation_data=(XTest, yTest))"""

PLOT_ACC = """\
f, ax = plt.subplots()
ax.plot(historico.history['accuracy'], 'o-')
ax.plot(historico.history['val_accuracy'], 'x-')
ax.legend(['Acurácia no Treinamento', 'Acurácia na Validação'], loc=0)
ax.set_title('Treinamento e Validação - Acurácia por Época')
ax.set_xlabel('Época')
ax.set_ylabel('Acurácia')"""

MATRIZ = """\
# Matriz de confusão no conjunto de teste
yPred = np.argmax(model.predict(XTest), axis=1)
cm = confusion_matrix(yTestLabels, yPred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title(f'Matriz de Confusão - {model.name}')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

print(classification_report(yTestLabels, yPred, digits=4))"""


# --- Definição das arquiteturas (texto que vai para a célula de código) ---
LENET = """\
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

inputShape = (imageRows, imageCols, cores)
model = Sequential(name='LeNet')
model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(50, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(numClasses))
model.add(Activation('softmax'))
model.summary()"""

CNN = """\
from keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, Dropout,
                          Flatten, Dense)

inputShape = (imageRows, imageCols, cores)
model = Sequential(name='CNN_Moderna')

# Bloco 1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Bloco 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classificador
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='softmax'))
model.summary()"""

MLP = """\
from keras.layers import Dense, Dropout

entrada = imageRows * imageCols  # 784
model = Sequential(name='MLP')
model.add(Dense(512, activation='relu', input_shape=(entrada,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(numClasses, activation='softmax'))
model.summary()"""


def badge_colab(nome_arquivo):
    url = f"https://colab.research.google.com/github/{REPO_GITHUB}/blob/main/{nome_arquivo}"
    return (f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({url})\n\n> 💡 **Dica:** rode no Google Colab com GPU "
            f"(*Ambiente de execução → Alterar o tipo → GPU*) para treinar em poucos minutos.")


def gerar(nome_arquivo, titulo, intro, arquitetura, achatar):
    cells = [
        md(titulo + "\n\n" + intro + "\n\n" + badge_colab(nome_arquivo)),
        md("## 0. Setup do ambiente"),
        code(SETUP_COLAB),
        md("## 1. Importações"),
        code(IMPORTS),
        md("## 2. Hiperparâmetros"),
        code(HIPER),
        md("## 3. Carga e pré-processamento do MNIST"),
        code(bloco_dados(achatar)),
        md("## 4. Definição da arquitetura"),
        code(arquitetura),
        md("## 5. Compilação e treino (10 épocas)"),
        code(TREINO),
        md("## 6. Curva de acurácia"),
        code(PLOT_ACC),
        md("## 7. Matriz de confusão e relatório de classificação"),
        code(MATRIZ),
    ]
    caminho = os.path.join(RAIZ, nome_arquivo)
    with open(caminho, "w") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
    print("gerado:", nome_arquivo)


def gerar_comparacao():
    cells = [
        md("# 📊 Comparação das Três Soluções — MNIST\n\n"
           "Notebook **autossuficiente**: treina as três arquiteturas (**LeNet**, "
           "**CNN Moderna** e **MLP**) por 10 épocas e compara as matrizes de confusão "
           "lado a lado. É só executar todas as células (no Colab, *Ambiente de "
           "execução → Executar tudo*).\n\n" + badge_colab("4_Comparacao.ipynb")),
        md("## 0. Setup do ambiente"),
        code(SETUP_COLAB),
        md("## 1. Importações e dados"),
        code("""\
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import utils as utls
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Activation, BatchNormalization,
                          Dropout, Flatten, Dense)
from sklearn.metrics import confusion_matrix

tf.random.set_seed(42); np.random.seed(42)

numClasses, batchSize, epochs = 10, 256, 10
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
XTrain = XTrain.astype('float32') / 255.0
XTest = XTest.astype('float32') / 255.0
yTestLabels = yTest.copy()
yTrainCat = utls.to_categorical(yTrain, numClasses)
yTestCat = utls.to_categorical(yTest, numClasses)

# Versões 2D (CNNs) e achatada (MLP)
XTrain2D, XTest2D = XTrain.reshape(-1, 28, 28, 1), XTest.reshape(-1, 28, 28, 1)
XTrainFlat, XTestFlat = XTrain.reshape(-1, 784), XTest.reshape(-1, 784)"""),
        md("## 2. Definição das três arquiteturas"),
        code("""\
def criar_lenet():
    m = Sequential(name='LeNet')
    m.add(Conv2D(20, (5, 5), padding='same', input_shape=(28, 28, 1))); m.add(Activation('relu'))
    m.add(MaxPooling2D((2, 2), strides=2))
    m.add(Conv2D(50, (5, 5), padding='same')); m.add(Activation('relu'))
    m.add(MaxPooling2D((2, 2), strides=2))
    m.add(Flatten()); m.add(Dense(500)); m.add(Activation('relu'))
    m.add(Dense(numClasses)); m.add(Activation('softmax'))
    return m

def criar_cnn_moderna():
    m = Sequential(name='CNN_Moderna')
    m.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    m.add(BatchNormalization())
    m.add(Conv2D(32, (3, 3), padding='same', activation='relu')); m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2))); m.add(Dropout(0.25))
    m.add(Conv2D(64, (3, 3), padding='same', activation='relu')); m.add(BatchNormalization())
    m.add(Conv2D(64, (3, 3), padding='same', activation='relu')); m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2))); m.add(Dropout(0.25))
    m.add(Flatten()); m.add(Dense(256, activation='relu')); m.add(BatchNormalization())
    m.add(Dropout(0.5)); m.add(Dense(numClasses, activation='softmax'))
    return m

def criar_mlp():
    m = Sequential(name='MLP')
    m.add(Dense(512, activation='relu', input_shape=(784,))); m.add(Dropout(0.2))
    m.add(Dense(256, activation='relu')); m.add(Dropout(0.2))
    m.add(Dense(numClasses, activation='softmax'))
    return m"""),
        md("## 3. Treino das três redes (10 épocas cada)"),
        code("""\
configs = [
    ('LeNet',       criar_lenet(),       XTrain2D,   XTest2D),
    ('CNN_Moderna', criar_cnn_moderna(), XTrain2D,   XTest2D),
    ('MLP',         criar_mlp(),         XTrainFlat, XTestFlat),
]

resultados = []
for nome, modelo, Xtr, Xte in configs:
    print(f'\\n===== Treinando {nome} =====')
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = modelo.fit(Xtr, yTrainCat, batch_size=batchSize, epochs=epochs,
                      validation_data=(Xte, yTestCat), verbose=2)
    yPred = np.argmax(modelo.predict(Xte, verbose=0), axis=1)
    cm = confusion_matrix(yTestLabels, yPred)
    acc = float((yPred == yTestLabels).mean())
    resultados.append({'nome': nome, 'acc': acc, 'params': modelo.count_params(),
                       'cm': cm, 'val': hist.history['val_accuracy']})
    print(f'{nome}: acurácia de teste = {acc*100:.2f}%')"""),
        md("## 4. Tabela-resumo"),
        code("""\
print(f"{'Arquitetura':<14} {'Parâmetros':>12} {'Acurácia':>10} {'Erros':>7}")
for r in resultados:
    erros = int(r['cm'].sum() - np.trace(r['cm']))
    print(f"{r['nome']:<14} {r['params']:>12,} {r['acc']*100:>9.2f}% {erros:>7}")"""),
        md("## 5. Matrizes de confusão lado a lado"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, r in zip(axes, resultados):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, square=True)
    ax.set_title(f"{r['nome']}\\nAcurácia: {r['acc']*100:.2f}%")
    ax.set_xlabel('Predito'); ax.set_ylabel('Real')
fig.suptitle('Matrizes de Confusão - MNIST (10 épocas)', fontsize=16, y=1.03)
plt.tight_layout(); plt.show()"""),
        md("## 6. Acurácia de validação por época"),
        code("""\
plt.figure(figsize=(9, 5))
for r in resultados:
    plt.plot(range(1, len(r['val']) + 1), r['val'], 'o-', label=r['nome'])
plt.title('Acurácia de Validação por Época')
plt.xlabel('Época'); plt.ylabel('Acurácia (validação)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()"""),
        md("## 7. Conclusão\n\n"
           "_Veja a análise textual completa no `README.md` da raiz do projeto "
           "(seção **Resultados e Conclusão**)._"),
    ]
    caminho = os.path.join(RAIZ, "4_Comparacao.ipynb")
    with open(caminho, "w") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
    print("gerado: 4_Comparacao.ipynb")


if __name__ == "__main__":
    gerar(
        "1_LeNet.ipynb",
        "# 🧠 Arquitetura 1 — LeNet (MNIST)",
        "Implementação da **LeNet** baseada no exemplo `marceloarantes19/deepLearning`. "
        "Rede convolucional clássica (LeCun, 1998): dois blocos Conv+ReLU+MaxPooling "
        "seguidos de uma camada densa de 500 neurônios e saída softmax de 10 classes.",
        LENET, achatar=False,
    )
    gerar(
        "2_CNN_Moderna.ipynb",
        "# 🚀 Arquitetura 2 — CNN Moderna (BatchNorm + Dropout)",
        "CNN profunda no estilo **VGG simplificado**: blocos de convoluções 3x3 "
        "empilhadas, **BatchNormalization** para estabilizar o treino e **Dropout** "
        "para regularização. Mais profunda e robusta que a LeNet.",
        CNN, achatar=False,
    )
    gerar(
        "3_MLP.ipynb",
        "# 🔢 Arquitetura 3 — MLP (Perceptron Multicamadas)",
        "Rede totalmente conectada que **ignora a estrutura espacial** da imagem: os "
        "784 pixels viram um vetor de entrada. Serve de linha de base para evidenciar "
        "o ganho das convoluções em problemas de visão.",
        MLP, achatar=True,
    )
    gerar_comparacao()
