# 1. Instalar e carregar os pacotes necessários
cat("\014")  # Limpa o console
required_packages <- c("rminer", "smotefamily", "PRROC", "beepr", 
                       "caret", "pdp", "ggplot2", "doParallel", "foreach", "kernlab")
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

# 5. Configurar paralelização
total_cores <- detectCores()
cl <- makeCluster(floor(total_cores * 0.75))  # Usa 75% dos núcleos disponíveis
registerDoParallel(cl)
cat("Núcleos configurados para paralelização:", floor(total_cores * 0.75), "\n")

# 6. Ajustar o modelo SVM com kernel linear e custo reduzido
cat("\nTreinando modelo com SVM (kernel linear)...\n")
svm_model <- ksvm(
  class ~ ., 
  data = train_data, 
  kernel = "vanilladot",  # Kernel linear
  C = 0.1,                # Parâmetro de custo reduzido
  prob.model = TRUE,      # Habilitar probabilidades
  scaled = TRUE,          # Escalar os dados
  threads = detectCores() # Máximo de threads suportado
)
cat("Modelo SVM (kernlab) treinado com sucesso!\n")

# Finalizar o cluster de paralelização
stopCluster(cl)

# 7. Realizar previsões
cat("\nRealizando previsões...\n")
predictions <- predict(svm_model, test_data, type = "probabilities")

# Criar dataframe de resultados
resultados <- data.frame(
  Registro = seq_along(test_data$class),
  Valor_Real = test_data$class,
  Predicao_Probabilidade = predictions[, 2],  # Probabilidade da classe 1
  Predicao_Classe = ifelse(predictions[, 2] >= 0.5, 1, 0)
)
cat("\nResultados iniciais:\n")
print(head(resultados))

# 8. Avaliar o modelo
cat("\nCalculando métricas de desempenho...\n")
confusion_matrix <- table(Valor_Real = resultados$Valor_Real, Predicao_Classe = resultados$Predicao_Classe)
cat("Matriz de Confusão:\n")
print(confusion_matrix)

precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precisão:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1_score, 3), "\n")

# 9. Calcular o AUC
cat("\nCalculando AUC...\n")
pred <- prediction(predictions[, 2], as.numeric(as.character(test_data$class)))
auc <- performance(pred, measure = "auc")@y.values[[1]]
cat("AUC:", round(auc, 3), "\n")

# 10. Exportar resultados
write.csv(resultados, "d:/DADOSgraficos_R/svm_results.csv", row.names = FALSE)

cat("\nProcesso concluído!\n")
beep(3)


# excluir arquivo png aberto no RSTUDIO 
#dev.list()
#dev.off(3)