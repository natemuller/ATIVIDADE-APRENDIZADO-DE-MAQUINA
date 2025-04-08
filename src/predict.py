import pandas as pd
import joblib

# Carregar modelo e scaler
knn = joblib.load("src/kaggle_comp_1/knn_model.pkl")
scaler = joblib.load("src/kaggle_comp_1/scaler.pkl")

# Carregar conjunto de teste
df_test = pd.read_csv("src/kaggle_comp_1/test.csv")

# Garantir que as colunas usadas no treino estão no teste
train_columns = joblib.load("src/kaggle_comp_1/train_columns.pkl")  # Salvar colunas no treinamento
X_test = df_test[train_columns]  # Seleciona apenas as colunas do treino

# Normalizar os dados de teste
X_test = scaler.transform(X_test)

# Fazer previsões
predictions = knn.predict(X_test)

# Criar arquivo de submissão
def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({'id': test_df.index, 'Target': predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file '{submission_file_name}' created successfully.")

create_submission_file(predictions, df_test, "src/kaggle_comp_1/submission.csv")

# Opcional: Salvar CSV com previsões
df_test["predictions"] = predictions
df_test.to_csv("src/kaggle_comp_1/resultado.csv", index=False)
print("Arquivo 'resultado.csv' criado com previsões.")

def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({'id': test_df.index, 'Target': predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file '{submission_file_name}' created successfully.")

create_submission_file(predictions, df_test, "submission.csv")

