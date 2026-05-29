"""Gera a comparação final entre as três soluções (LeNet, CNN Moderna e MLP).

Lê os artefatos salvos em `resultados/` por cada script de treino e produz:
- um painel com as três matrizes de confusão lado a lado (`matrizes_confusao.png`);
- um gráfico de acurácia de validação por época (`acuracia_por_epoca.png`);
- uma tabela-resumo em Markdown no terminal e em `resumo.md`.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados")
MODELOS = ["LeNet", "CNN_Moderna", "MLP"]


def carregar(nome):
    with open(os.path.join(RESULTS_DIR, f"{nome}.json")) as f:
        return json.load(f)


def plot_matrizes(dados):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, d in zip(axes, dados):
        cm = np.array(d["matriz_confusao"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=ax, square=True)
        ax.set_title(f"{d['nome']}\nAcurácia: {d['acuracia_teste']*100:.2f}%")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    fig.suptitle("Matrizes de Confusão — MNIST (10 épocas)", fontsize=16, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "matrizes_confusao.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_acuracia(dados):
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in dados:
        val = d["history"]["val_accuracy"]
        ax.plot(range(1, len(val) + 1), val, "o-", label=d["nome"])
    ax.set_title("Acurácia de Validação por Época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (validação)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "acuracia_por_epoca.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def erros_por_classe(cm):
    """Total de erros (fora da diagonal) por dígito real."""
    cm = np.array(cm)
    return {i: int(cm[i].sum() - cm[i, i]) for i in range(10)}


def tabela_resumo(dados):
    linhas = [
        "| Arquitetura | Parâmetros | Acurácia (teste) | Erros totais |",
        "|---|---|---|---|",
    ]
    for d in dados:
        cm = np.array(d["matriz_confusao"])
        erros = int(cm.sum() - np.trace(cm))
        linhas.append(
            f"| {d['nome']} | {d['parametros']:,} | "
            f"{d['acuracia_teste']*100:.2f}% | {erros} |"
        )
    return "\n".join(linhas)


def main():
    dados = [carregar(m) for m in MODELOS]
    plot_matrizes(dados)
    plot_acuracia(dados)

    tabela = tabela_resumo(dados)
    print("\n=== RESUMO COMPARATIVO ===\n")
    print(tabela)
    print("\n=== ERROS POR DÍGITO (fora da diagonal) ===\n")
    for d in dados:
        print(f"{d['nome']}: {erros_por_classe(d['matriz_confusao'])}")

    with open(os.path.join(RESULTS_DIR, "resumo.md"), "w") as f:
        f.write("# Resumo Comparativo\n\n")
        f.write(tabela + "\n\n")
        f.write("## Erros por dígito (fora da diagonal)\n\n")
        for d in dados:
            f.write(f"- **{d['nome']}**: {erros_por_classe(d['matriz_confusao'])}\n")
    print(f"\nArtefatos salvos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
