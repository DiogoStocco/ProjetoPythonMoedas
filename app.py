import joblib
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import pandas as pd

def clean_and_validate_data(file_path):
    """
    Limpa e valida os dados do arquivo CSV.
    Retorna um DataFrame limpo e verifica se há ao menos duas colunas numéricas.
    """
    data = pd.read_csv(file_path)

    # Remoção de linhas completamente nulas
    data = data.dropna(how='all')

    # Tentar converter todas as colunas para numérico (ignorar erros)
    # for col in data.columns:
    #     data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remover colunas que ficaram completamente nulas
    data = data.dropna(axis=1, how='all')

    # Verificar se há ao menos duas colunas numéricas
    if data.shape[1] < 2:
        raise ValueError("O arquivo precisa ter pelo menos duas colunas numéricas após limpeza.")

    # Preenchendo valores nulos com a média das colunas
    data = data.fillna(data.mean(numeric_only=True))

    # Verificar se ainda existem NaNs
    if data.isna().any().any():
        raise ValueError("Não foi possível limpar completamente os dados. Verifique o arquivo de entrada.")

    return data

# Página inicial: upload de arquivo
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for("analyze_file", filename=file.filename))
        else:
            return "Por favor, envie um arquivo CSV válido.", 400
    return render_template("upload.html")

# Página de análise
@app.route("/analyze/<filename>", methods=["GET", "POST"])
def analyze_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()

        if request.method == "POST":
            graph_type = request.form.get("graph_type")
            x_column = request.form.get("x_column")
            y_column = request.form.get("y_column")

            # Gerar gráfico com base na seleção do usuário
            if graph_type == "scatter" and x_column and y_column:
                fig = px.scatter(df, x=x_column, y=y_column, title="Scatter Plot")
            elif graph_type == "line" and x_column and y_column:
                fig = px.line(df, x=x_column, y=y_column, title="Gráfico de Linha")
            elif graph_type == "histogram" and x_column:
                fig = px.histogram(df, x=x_column, title="Histograma")
            else:
                return "Selecione os parâmetros corretos para o gráfico.", 400

            graph_html = fig.to_html(full_html=False)
        else:
            # Gráfico inicial padrão
            fig = px.histogram(df, x=columns[1], title="Histograma Inicial")
            graph_html = fig.to_html(full_html=False)

        return render_template(
            "analyze.html", graph_html=graph_html, columns=columns, filename=filename
        )
    except Exception as e:
        return f"Erro ao processar o arquivo: {e}", 500

# Rota para treinamento do modelo
@app.route("/train/<filename>", methods=["GET", "POST"])
def train_model(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = clean_and_validate_data(file_path)
        columns = df.columns.tolist()
        error_message = None

        if request.method == "POST":
            # Obter configurações do formulário
            x_columns = request.form.getlist("x_columns")
            y_column = request.form.get("y_column")
            model_type = request.form.get("model_type")
            param_k = int(request.form.get("param_k", 3))

            if not x_columns or not y_column:
                error_message = "Selecione ao menos uma coluna para X e uma para y."
            else:
                # Separar dados
                X = df[x_columns]
                y = df[y_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Treinamento do modelo
                if model_type == "linear":
                    model = LinearRegression()
                elif model_type == "knn":
                    model = KNeighborsRegressor(n_neighbors=param_k)
                else:
                    return "Modelo desconhecido.", 400

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                # Salvar modelo
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                model_filename = f"{model_type}_model_{timestamp}.joblib"
                model_path = os.path.join("models", model_filename)
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)

                return render_template(
                    "train.html",
                    columns=columns,
                    filename=filename,
                    mse=mse,
                    model_type=model_type,
                    x_columns=x_columns,
                    y_column=y_column,
                    model_filename=model_filename,
                )

        return render_template("train.html", columns=columns, filename=filename, error_message=error_message)
    except Exception as e:
        return f"Erro ao processar o arquivo: {e}", 500
    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()
        model_filename = request_data.get("model_filename")
        input_data = request_data.get("input_data")

        if not model_filename or not input_data:
            return {"error": "Forneça o nome do modelo e os dados de entrada."}, 400

        # Carregar modelo salvo
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            return {"error": "Modelo não encontrado."}, 404

        model = joblib.load(model_path)

        # Fazer predição
        input_df = pd.DataFrame([input_data])
        predictions = model.predict(input_df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True)
