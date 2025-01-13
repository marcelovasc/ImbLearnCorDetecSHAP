# Limpar o ambiente
rm(list = ls())
cat("\014")

# 1. Instalar e carregar os pacotes necessários
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

# 5. Ajustar o modelo (Redes Neurais com paralelização)
cat("\nTreinando modelo com redes neurais...\n")
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

nn_model <- fit(
  class ~ ., 
  data = train_data, 
  model = "mlp",  # Redes neurais do rminer
  task = "class"
)

stopCluster(cl)
cat("Modelo de redes neurais treinado com sucesso!\n")

# Criar método de predição compatível com pdp
predict_custom <- function(object, newdata) {
  as.numeric(as.character(predict(object, newdata)))
}

# 6. Realizar previsões
cat("\nRealizando previsões...\n")
predictions <- as.numeric(as.character(predict(nn_model, test_data[, -ncol(test_data)])))
predictions_binarias <- ifelse(predictions >= 0.5, 1, 0)

# Criar dataframe de resultados
resultados <- data.frame(
  Registro = seq_along(test_data$class),
  Valor_Real = as.numeric(as.character(test_data$class)),  # Corrigido
  Predicao_Probabilidade = predictions,
  Predicao_Classe = predictions_binarias
)
cat("\nResultados iniciais:\n")
print(head(resultados))

# 7. Avaliar o modelo
cat("\nCalculando métricas de desempenho...\n")
confusion_matrix <- table(Valor_Real = resultados$Valor_Real, Predicao_Classe = resultados$Predicao_Classe)
cat("Matriz de Confusão:\n")
print(confusion_matrix)

# Evitar divisão por zero
if (sum(confusion_matrix[, 2]) > 0 && sum(confusion_matrix[2, ]) > 0) {
  precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  f1_score <- 2 * (precision * recall) / (precision + recall)
} else {
  precision <- recall <- f1_score <- NA
}

cat("Precisão:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1_score, 3), "\n")

# 8. Gráficos de importância das variáveis
cat("\nGerando gráficos de importância das variáveis...\n")
importance <- Importance(nn_model, data = train_data)

# Exportação das informações de importância
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
    title = "Importância das Variáveis (1D-SA)",
    x = "Variáveis",
    y = "Importância"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))
dev.off()

cat("\nGráficos de importância gerados com sucesso!\n")

# 9. Curva Precision-Recall
cat("\nPlotando curva Precision-Recall...\n")
pr <- pr.curve(
  scores.class0 = predictions,
  weights.class0 = as.numeric(resultados$Valor_Real == 1),
  curve = TRUE
)
png(filename = "d:/DADOSgraficos_R/Precision-Recall_Curve.png", width = 800, height = 600)
plot(pr, main = "Precision-Recall Curve", auc.main = TRUE)
dev.off()

# 10. Análise de Sensibilidade (PDP)
cat("\nGerando gráficos PDP para variáveis mais importantes...\n")
for (var in importance$dnames[1:3]) {
  pd <- tryCatch(
    partial(
      object = nn_model,
      pred.var = var,
      train = train_data,
      pred.fun = predict_custom # Usar o método customizado
    ),
    error = function(e) {
      cat(sprintf("\nErro ao gerar PDP para a variável '%s': %s\n", var, e$message))
      return(NULL)
    }
  )
  
  if (!is.null(pd)) {
    png(filename = paste0("d:/DADOSgraficos_R/PDP_", var, ".png"), width = 800, height = 600)
    plotPartial(pd, main = paste("Perfil Parcial para", var))
    dev.off()
  }
}

cat("\nProcesso concluído!\n")
beep(3)

# Calcular AUC usando o PRROC
cat("\nCalculando AUC...\n")
pr_curve <- roc.curve(
  scores.class0 = resultados$Predicao_Probabilidade[resultados$Valor_Real == 1],
  scores.class1 = resultados$Predicao_Probabilidade[resultados$Valor_Real == 0],
  curve = FALSE # Define que não haverá geração da curva
)
cat(sprintf("AUC: %.3f\n", pr_curve$auc))

