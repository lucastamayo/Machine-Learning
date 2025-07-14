# Machine-Learning
Repositório criado para o projeto final da disciplina de Aprendizagem de Máquinas PPGF-UERJ

### Modelos BDT (Boosted Decision Trees)

- `train.py`  
  Treina o modelo BDT utilizando o pacote xgboost com validação Cross-Validation, usando pesos por evento.
- `trainvstest.py`  
  Compara o desempenho da BDT nos conjuntos de treino e teste, gerando gráficos relativos à curva ROC e PR (Precision Recall)

- `metricas.py`  
  Plota métricas de avaliação (AUC, matriz de confusão, etc.) usando apenas o conjunto de teste do BDT treinado.

### Modelos MLP (Multi-Layer Perceptron)

- `train_MLP.py`  
  Treina um classificador Multilayer Perceptron (MLP) com divisão treino/teste e armazenamento dos resultados. Também utiliza Cross-Validation

- `metricas_mlp.py`  
  Gera gráficos comparativos de métricas de avaliação para o modelo MLP.
