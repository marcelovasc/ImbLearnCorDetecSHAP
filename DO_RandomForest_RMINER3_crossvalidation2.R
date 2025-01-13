# 1. Instalar e carregar os pacotes necessários
cat("\014")  # Limpa o console
required_packages <- c("rminer", "smotefamily", "PRROC", "beepr", 
                       "caret", "pdp", "ggplot2", "doParallel", "foreach")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# 2. Carregar os dados
file_path <- "d:/DADOS/DFFull_Final2.csv"
df <- read.csv(file_path)

cat("Estrutura do dataset:\n")
str(df)
cat("\nDistribuição inicial da variável TARGET_CORRUPCAO:\n")
print(table(df$TARGET_CORRUPCAO))

# Remover coluna de autonumeração X, se existir
df <- df[, !names(df) %in% "X"]

# Converter TARGET_CORRUPCAO em fator
df$TARGET_CORRUPCAO <- as.factor(df$TARGET_CORRUPCAO)

cat("Estrutura após ajustes:\n")
str(df)

# 3. Aplicar SMOTE para balancear as classes
set.seed(42)
df_balanced <- SMOTE(
  df[, -which(names(df) == "TARGET_CORRUPCAO")],
  df$TARGET_CORRUPCAO,
  K = 5,
  dup_size = 0
)

balanced_data <- df_balanced$data
balanced_data$class <- as.factor(balanced_data$class)

cat("Distribuição após SMOTE:\n")
print(table(balanced_data$class))

# 4. Dividir os dados em treino e teste
set.seed(123)
train_index <- createDataPartition(balanced_data$class, p = 0.7, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

cat("\nDistribuição Dados de Treino:\n")
print(table(train_data$class))
cat("\nDistribuição Dados de Teste:\n")
print(table(test_data$class))

# 5. Ajustar o modelo Random Forest com paralelização
cat("\nTreinando modelo com Random Forest...\n")
cl <- makeCluster(detectCores() - 1)  # Usa todos os núcleos menos um
registerDoParallel(cl)

random_forest_model <- fit(
  class ~ ., 
  data = train_data, 
  model = "randomforest"
)

stopCluster(cl)
cat("Modelo Random Forest treinado com sucesso!\n")

# 6. Realizar previsões
cat("\nRealizando previsões...\n")
# Retorna as probabilidades das classes
predictions_prob <- predict(random_forest_model, test_data, type = "prob")

# Converter probabilidades para classes binárias
predictions_class <- ifelse(predictions_prob[, 2] >= 0.5, 1, 0)

# Criar dataframe de resultados
resultados <- data.frame(
  Registro = seq_along(test_data$class),
  Valor_Real = as.numeric(as.character(test_data$class)),  # Converter para valores numéricos
  Predicao_Probabilidade = predictions_prob[, 2],
  Predicao_Classe = predictions_class
)
cat("\nResultados iniciais:\n")
print(head(resultados))

# 7. Avaliar o modelo
cat("\nCalculando métricas de desempenho...\n")
confusion_matrix <- table(Valor_Real = resultados$Valor_Real, Predicao_Classe = resultados$Predicao_Classe)
cat("Matriz de Confusão:\n")
print(confusion_matrix)

# Calcular precisão, recall e F1-Score
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precisão:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1_score, 3), "\n")

# Calcular AUC
cat("\nCalculando AUC...\n")
pr <- roc.curve(
  scores.class0 = resultados$Predicao_Probabilidade[resultados$Valor_Real == 1],
  scores.class1 = resultados$Predicao_Probabilidade[resultados$Valor_Real == 0],
  curve = FALSE
)
cat(sprintf("AUC: %.3f\n", pr$auc))

# 8. Gráficos de importância das variáveis
cat("\nGerando gráficos de importância das variáveis...\n")
importance <- Importance(random_forest_model, data = train_data)

# Exportar importâncias para CSV
write.csv(data.frame(Variable = importance$dnames, Importance = importance$imp), 
          "d:/DADOSgraficos_R/importance_values.csv", row.names = FALSE)

# Gráfico com ggplot2
importance_data <- data.frame(
  Variable = importance$dnames,
  Importance = importance$imp
)
png(filename = "d:/DADOSgraficos_R/Importance_GGPlot.png", width = 800, height = 600)
ggplot(importance_data, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(
    title = "Importância das Variáveis (Random Forest)",
    x = "Variáveis",
    y = "Importância"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))
dev.off()

# 9. Conclusão
cat("\nProcesso concluído!\n")
beep(3)


# excluir arquivo png aberto no RSTUDIO 
#dev.list()
#dev.off(3)
