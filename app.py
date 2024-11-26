from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Configurações iniciais
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variáveis globais
latest_csv_file = None
model = None


def train_model(file_path):
    global model

    # Leitura dos dados
    data = pd.read_csv(file_path)

    # Remoção de linhas completamente nulas
    data = data.dropna(how='all')

    # Tentar converter todas as colunas para numérico (ignorar erros)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remover colunas que ficaram completamente nulas
    data = data.dropna(axis=1, how='all')

    # Verificar se há ao menos duas colunas numéricas
    if data.shape[1] < 2:
        raise ValueError("O arquivo precisa ter pelo menos duas colunas numéricas após limpeza")

    # Preenchendo valores nulos com a média e convertendo para inteiro
    data = data.fillna(data.mean()).astype(int)

    # Separação de features e target
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]

    # Treinamento do modelo
    model = LinearRegression()
    model.fit(features, target)

    # Salvando o modelo treinado
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    joblib.dump(model, model_path)

    print("Modelo treinado com sucesso! Dados convertidos para inteiros.")

def load_model():
    global model
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    print(f"Tentando carregar o modelo do caminho: {model_path}")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Modelo carregado com sucesso.")
    else:
        print("Modelo não encontrado. Criando um modelo padrão.")

        # Criar um modelo padrão (LinearRegression vazio)
        model = LinearRegression()

        # Salvar o modelo no arquivo model.pkl
        try:
            joblib.dump(model, model_path)
            print(f"Modelo padrão criado e salvo em '{model_path}'.")
        except Exception as e:
            print(f"Erro ao salvar o modelo padrão: {e}")


# Página inicial
@app.route('/')
def index():
    return render_template('index.html')


# Rota para upload do CSV
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global latest_csv_file

    if request.method == 'POST':
        if 'csvFile' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files['csvFile']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Por favor, envie um arquivo CSV válido"}), 400

        # Salvar o arquivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        latest_csv_file = file.filename

        # Treinamento automático após o upload
        try:
            train_model(file_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Erro ao treinar o modelo: {str(e)}"}), 500

        return redirect(url_for('visualize'))

    return render_template('upload.html')


# Rota para visualização e análise de dados
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    global latest_csv_file

    if not latest_csv_file:
        return jsonify({"error": "Nenhum arquivo CSV disponível"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_csv_file)
    data = pd.read_csv(file_path)

    if request.method == 'POST':
        column_x = request.form.get('column_x')
        column_y = request.form.get('column_y')
        if column_x and column_y:
            fig = px.scatter(data, x=column_x, y=column_y, title=f'Relação entre {column_x} e {column_y}')
            graph_html = fig.to_html(full_html=False)
            return render_template('visualize.html', data=data.head(), graph=graph_html)

    return render_template('visualize.html', data=data.head(), graph=None)


# Rota para predições
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, latest_csv_file

    if model is None:
        return jsonify({"error": "Modelo ainda não treinado"}), 400

    if not latest_csv_file:
        return jsonify({"error": "Nenhum arquivo CSV disponível"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_csv_file)
    data = pd.read_csv(file_path)

    # Converter tudo para numérico e preencher valores nulos
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(axis=1, how='all').fillna(data.mean())

    if request.method == 'POST':
        target_column = request.form.get('target_column')  # Coluna a ser prevista

        if not target_column or target_column not in data.columns:
            return jsonify({"error": "Coluna-alvo inválida"}), 400

        # Separar features e alvo
        features = data.drop(columns=[target_column])
        if features.empty:
            return jsonify({"error": "Não há colunas suficientes para usar como features"}), 400

        # Calcular previsões para 5, 10, 15 e 50 anos
        predictions = {}
        years_to_predict = [5, 10, 15, 50]
        try:
            for year in years_to_predict:
                # Criar um conjunto de features fictícias ajustando os valores atuais com base nos anos
                input_features = features.mean(axis=0) * year
                prediction = model.predict([input_features])[0]

                # Formatando o valor como um inteiro (dólares)
                predictions[f"{year} anos"] = f"${int(round(prediction))}"

            return jsonify({
                "message": f"Previsão realizada com sucesso para a coluna '{target_column}'",
                "predictions": predictions
            })
        except Exception as e:
            return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500

    # Preparar a lista de colunas disponíveis para seleção no formulário
    columns = data.columns.tolist()
    return render_template('predict.html', columns=columns)


if __name__ == '__main__':
    app.run(debug=True)
