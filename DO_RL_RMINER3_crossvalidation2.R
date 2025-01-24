# 1. Instalar e carregar os pacotes necessários
cat("\014")  # Limpa o console
required_packages <- c("rminer", "smotefamily", "PRROC", "beepr", 
                       "caret", "pdp", "ggplot2", "doParallel", "foreach", "pROC")
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

# 5. Ajustar o modelo (Regressão Logística com paralelização)
cat("\nTreinando modelo com regressão logística...\n")
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
logistic_model <- glm(
  class ~ ., 
  data = train_data,
  family = binomial
)
stopCluster(cl)
cat("Modelo de regressão logística treinado com sucesso!\n")
summary(logistic_model)

# 6. Realizar previsões
cat("\nRealizando previsões...\n")
predictions <- predict(logistic_model, newdata = test_data[, -ncol(test_data)], type = "response")
predictions_binarias <- ifelse(predictions >= 0.5, 1, 0)

# Criar dataframe de resultados
resultados <- data.frame(
  Registro = seq_along(test_data$class),
  Valor_Real = test_data$class,
  Predicao_Probabilidade = predictions,
  Predicao_Classe = predictions_binarias
)
cat("\nResultados iniciais:\n")
print(head(resultados))

# 7. Avaliar o modelo
cat("\nCalculando métricas de desempenho...\n")
confusion_matrix <- table(Valor_Real = test_data$class, Predicao_Classe = predictions_binarias)
cat("Matriz de Confusão:\n")
print(confusion_matrix)

precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precisão:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1_score, 3), "\n")

# 8. Gráficos de importância das variáveis
cat("\nGerando gráficos de importância das variáveis...\n")
importance <- Importance(logistic_model, data = train_data)

# Exportar as informações de IMPORTANCE
write.csv(data.frame(Variable = importance$dnames, Importance = importance$imp), 
          "d:/DADOSgraficos_R/importance_values.csv", row.names = FALSE)

# Gráfico básico de barras
png(filename = "d:/DADOSgraficos_R/Importance_Basic.png", width = 800, height = 600)
barplot(
  importance$imp, 
  main = "Importância das Variáveis (1D-SA)", 
  names.arg = importance$dnames, 
  las = 2, 
  col = "blue", 
  cex.names = 0.7
)
dev.off()

# Gráfico dos 20 principais atributos
top_20_indices <- order(importance$imp, decreasing = TRUE)[1:20]
top_20_importance <- importance$imp[top_20_indices]
top_20_names <- importance$dnames[top_20_indices]

png(filename = "d:/DADOSgraficos_R/Importance_Top20.png", width = 800, height = 600)
barplot(
  top_20_importance, 
  main = "Top 20 Variáveis Importantes (1D-SA)", 
  names.arg = top_20_names, 
  las = 2, 
  col = "blue", 
  cex.names = 0.7
)
dev.off()

# 9. Curva Precision-Recall
cat("\nPlotando curva Precision-Recall...\n")
pr <- pr.curve(
  scores.class0 = as.numeric(predictions),
  weights.class0 = as.numeric(test_data$class == 1),
  curve = TRUE
)

png(filename = "d:/DADOSgraficos_R/Precision-Recall_Curve.png", width = 800, height = 600)
plot(pr, main = "Precision-Recall Curve", auc.main = TRUE)
dev.off()

# 10. Análise de Sensibilidade (PDP)
cat("\nGerando gráficos PDP para variáveis mais importantes...\n")
for (var in top_20_names[1:3]) {
  if (!var %in% colnames(train_data)) {
    cat(sprintf("Skipping variable %s as it is not in the training data.\n", var))
    next
  }
  
  unique_values <- unique(train_data[[var]])
  if (length(unique_values) < 2) {
    cat(sprintf("Skipping variable %s as it has less than 2 unique values in the training data.\n", var))
    next
  }
  
  tryCatch({
    pd <- partial(
      object = logistic_model,
      pred.var = var,
      train = train_data
    )
    png(filename = paste0("d:/DADOSgraficos_R/PDP_", var, ".png"), width = 800, height = 600)
    plotPartial(pd, main = paste("Partial Dependence for", var))
    dev.off()
    cat(sprintf("Gráfico PDP gerado para a variável: %s\n", var))
  }, error = function(e) {
    cat(sprintf("Erro ao gerar PDP para %s: %s\n", var, e$message))
  })
}

cat("\nProcesso concluído!\n")
beep(3)


# Exibir avisos
cat("\nWarnings:\n")
warnings()


