# 🚀 Como rodar no Google Colab (com GPU)

O Colab oferece **GPU gratuita** e roda as três redes em poucos minutos (em vez de ~1 hora na CPU local).

## Opção A — Um clique (recomendado)

1. Abra o notebook **autossuficiente** de comparação clicando no badge:

   [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Felipe-Alcantara/Trabalho-2---Deep-Learning-aplicado-ao-MNist/blob/main/4_Comparacao.ipynb)

2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → Acelerador de hardware: GPU → Salvar**.
3. Rode tudo: **Ambiente de execução → Executar tudo** (ou `Ctrl+F9`).

Esse notebook treina as **três** arquiteturas (LeNet, CNN Moderna e MLP) por 10 épocas e já mostra a tabela-resumo, as **três matrizes de confusão lado a lado** e a curva de acurácia.

## Opção B — Um notebook por arquitetura

Se quiser rodar cada rede separadamente, abra individualmente:

| Notebook | Abrir no Colab |
|---|---|
| `1_LeNet.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Felipe-Alcantara/Trabalho-2---Deep-Learning-aplicado-ao-MNist/blob/main/1_LeNet.ipynb) |
| `2_CNN_Moderna.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Felipe-Alcantara/Trabalho-2---Deep-Learning-aplicado-ao-MNist/blob/main/2_CNN_Moderna.ipynb) |
| `3_MLP.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Felipe-Alcantara/Trabalho-2---Deep-Learning-aplicado-ao-MNist/blob/main/3_MLP.ipynb) |
| `4_Comparacao.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Felipe-Alcantara/Trabalho-2---Deep-Learning-aplicado-ao-MNist/blob/main/4_Comparacao.ipynb) |

> ⚠️ Os badges só funcionam **depois** que os notebooks estiverem no GitHub (branch `main`). Antes disso, use **Arquivo → Fazer upload de notebook** no Colab.

## Rodar localmente (CPU)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Treinar pelos scripts (salva resultados em solucoes/resultados/)
cd solucoes
python treinar_lenet.py
python treinar_cnn_moderna.py
python treinar_mlp.py
python comparar.py                 # gera os gráficos comparativos
```
