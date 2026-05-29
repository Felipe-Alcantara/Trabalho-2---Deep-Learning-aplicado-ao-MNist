# 🤖 Contexto Operacional do Projeto (IA.md)

> Memória técnica do projeto, no padrão `felixo-standards/core/IA.md`. Registra objetivo, stack, decisões e resultados para retomada rápida por outra sessão/IA.

---

## 🎯 OBJETIVO DO PROJETO

[2026-05-29] Trabalho 2 de Inteligência Artificial: resolver o **MNIST** com três arquiteturas de redes neurais profundas e comparar suas **matrizes de confusão**. Base: exemplo **LeNet** de `marceloarantes19/deepLearning`. Exigência do enunciado: pesquisar mais **duas** arquiteturas, descrevê-las (camadas, ativação, características), treiná-las por **10 épocas** como o exemplo, comparar as três matrizes de confusão e concluir.

---

## 🏁 METAS & MILESTONES

- [2026-05-29] ✅ Replicar a LeNet do exemplo em script + notebook.
- [2026-05-29] ✅ Implementar 2ª arquitetura: **CNN Moderna** (Conv 3x3 + BatchNorm + Dropout).
- [2026-05-29] ✅ Implementar 3ª arquitetura: **MLP** (rede totalmente conectada).
- [2026-05-29] ✅ Treinar as três por 10 épocas e salvar histórico + matriz de confusão.
- [2026-05-29] ✅ Gerar comparação (painel das 3 matrizes + curva de acurácia + tabela).
- [2026-05-29] ✅ Documentar no README (descrições, resultados, conclusão, link do GitHub).

---

## 🛠️ STACK & DEPENDÊNCIAS

[2026-05-29] Python 3.12 em `.venv` (virtualenv local, ignorado no git).
[2026-05-29] `tensorflow-cpu` (Keras embutido), `scikit-learn` (matriz de confusão e relatório), `matplotlib` + `seaborn` (gráficos), `jupyter` (notebooks). Ver `requirements.txt`.

---

## 📐 DECISÕES DE ARQUITETURA

[2026-05-29] **Separação de responsabilidades**: `solucoes/common.py` concentra carga de dados, treino e avaliação; cada `treinar_*.py` só define a arquitetura. Evita arquivo "faz-tudo" (Guia Mínimo de Qualidade, item 2).
[2026-05-29] **Scripts + notebooks**: o treino oficial roda pelos scripts (reprodutível, headless) e salva JSON em `solucoes/resultados/`. Os notebooks `1_..4_` são a entrega didática, gerados por `gerar_notebooks.py` a partir do mesmo código — evita divergência entre script e notebook.
[2026-05-29] **Hiperparâmetros idênticos ao exemplo** (batch 256, 10 épocas, otimizador adam, perda categorical_crossentropy) para que a comparação entre arquiteturas seja justa.
[2026-05-29] **Seed fixa (42)** em TensorFlow e NumPy para resultados reprodutíveis.

---

## 🎨 DECISÕES DE DESIGN & CONVENÇÕES

[2026-05-29] Código e comentários em **português** (pedido do usuário). Documentação no padrão `felixo-standards` (README com header/badges/índice/estrutura/autor/CTA; este IA.md).
[2026-05-29] `felixo-standards/` é referência vendorizada e fica no `.gitignore` (não faz parte da entrega).

---

## 🧪 TESTES IMPORTANTES

[2026-05-29] ✅ Execução real no **Google Colab (GPU)**, 10 épocas cada — resultados oficiais da entrega:
  - CNN Moderna: **99,44%** (56 erros) — melhor.
  - LeNet: **99,05%** (95 erros).
  - MLP: **98,29%** (171 erros).
  Artefatos em `resultados/` (matrizes de confusão, curva de acurácia, resumo).
[2026-05-29] ✅ Smoke test local (1 época) do `4_Comparacao.ipynb` rodou de ponta a ponta (exit 0), confirmando que o notebook executa sem erro de código.
[2026-05-29] OBSERVAÇÃO: a CNN Moderna fica ~11% na época 1 (BatchNorm ainda instável) e dispara a partir da época 2-3 — comportamento esperado, não é bug.

---

## 🐛 BUGS & FIXES RELEVANTES

[2026-05-29] Ambiente inicial não tinha TensorFlow/numpy/sklearn. FIX: criado `.venv` e instaladas as dependências de `requirements.txt`.

---

## 🔗 INTEGRAÇÕES & SERVIÇOS EXTERNOS

[2026-05-29] Dataset MNIST baixado via `keras.datasets.mnist` (cache local em `~/.keras`). Sem credenciais.
[2026-05-29] Repositório GitHub da equipe: _preencher o link no README e aqui quando o remote for configurado._

---

## 📝 NOTAS GERAIS

[2026-05-29] Treino em CPU (`tensorflow-cpu`); a CNN Moderna é a mais lenta por época por ser a mais profunda.
