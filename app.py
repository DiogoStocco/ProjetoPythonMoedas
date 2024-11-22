from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import json

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
model_params = None

# Função para carregar o modelo
def load_model():
    global model
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    print(f"Tentando carregar o modelo do caminho: {model_path}")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Modelo carregado com sucesso.")
    else:
        model = None
        print("Modelo não encontrado.")
# Função para carregar os parâmetros do modelo
def load_model_params():
    global model_params
    model_params_path = 'model_params.json'
    try:
        if os.path.exists(model_params_path):
            with open(model_params_path, 'r') as f:
                model_params = json.load(f)
            print(f"Parâmetros do modelo carregados de '{model_params_path}'.")
        else:
            model_params = None  # Caso os parâmetros não existam ainda
            print(f"Arquivo '{model_params_path}' não encontrado. Parâmetros não carregados.")
    except Exception as e:
        model_params = None
        print(f"Erro ao carregar os parâmetros do modelo: {e}")

# Função para salvar os parâmetros do modelo
def save_model_params(params):
    model_params_path = 'model_params.json'
    try:
        with open(model_params_path, 'w') as f:
            json.dump(params, f)
        print(f"Parâmetros do modelo salvos em '{model_params_path}'.")
    except Exception as e:
        print(f"Erro ao salvar os parâmetros do modelo: {e}")

def get_all_unique_columns():
    upload_folder = app.config['UPLOAD_FOLDER']
    unique_columns = set()
    for filename in os.listdir(upload_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(upload_folder, filename)
            df = pd.read_csv(file_path)
            unique_columns.update(df.columns)
    return sorted(unique_columns)

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

# Rota para treinar o modelo
@app.route('/train', methods=['GET', 'POST'])
def train():
    global model_params  # Declarar que vamos usar a variável global model_params

    print("Iniciando o treinamento do modelo.")

    upload_folder = app.config['UPLOAD_FOLDER']
    data_list = []

    # Ler os arquivos CSV no diretório de uploads
    for filename in os.listdir(upload_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(upload_folder, filename)
            print(f"Lendo arquivo: {file_path}")
            try:
                df = pd.read_csv(file_path, thousands=',')
                print(f"Colunas no arquivo {filename}: {df.columns.tolist()}")
                data_list.append(df)
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")

    if not data_list:
        print("Nenhum arquivo CSV disponível para treinamento.")
        return jsonify({"error": "Nenhum arquivo CSV disponível para treinamento"}), 400

    # Concatenar todos os DataFrames
    try:
        data = pd.concat(data_list, ignore_index=True)
        print(f"Número total de linhas após concatenação: {len(data)}")
    except Exception as e:
        print(f"Erro ao concatenar os DataFrames: {e}")
        return jsonify({"error": f"Erro ao concatenar os arquivos CSV: {str(e)}"}), 500

    # Verificar se as colunas 'Date' e 'Price' existem
    if 'Date' not in data.columns or 'Price' not in data.columns:
        print("As colunas 'Date' e 'Price' são necessárias no CSV.")
        print(f"Colunas disponíveis: {data.columns.tolist()}")
        return jsonify({"error": "As colunas 'Date' e 'Price' são necessárias no CSV."}), 400

    # Converter a coluna 'Date' para datetime
    try:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce', infer_datetime_format=True)
        data = data.dropna(subset=['Date'])
        num_dates = data['Date'].notnull().sum()
        print(f"Número de datas válidas após conversão: {num_dates}")
    except Exception as e:
        print(f"Erro ao converter a coluna 'Date' para datetime: {e}")
        return jsonify({"error": f"Erro ao processar a coluna 'Date': {str(e)}"}), 500

    # Converter datas para valores numéricos (ordinal)
    try:
        data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    except Exception as e:
        print(f"Erro ao converter as datas para ordinal: {e}")
        return jsonify({"error": f"Erro ao converter as datas para ordinal: {str(e)}"}), 500

    # Selecionar features e target
    try:
        X = data[['Date_ordinal']]
        y = data['Price'].apply(lambda x: float(str(x).replace(',', '').replace(' ', '')))
        print(f"Exemplo de X:\n{X.head()}")
        print(f"Exemplo de y:\n{y.head()}")
    except Exception as e:
        print(f"Erro ao preparar as features e target: {e}")
        return jsonify({"error": f"Erro ao preparar as features e target: {str(e)}"}), 500

    # Treinamento do modelo
    try:
        model = LinearRegression()
        model.fit(X, y)
        print("Modelo treinado com sucesso.")

        # Extrair os coeficientes e intercepto
        coef = float(model.coef_[0])
        intercept = float(model.intercept_)
        model_params = {'coef': coef, 'intercept': intercept}

        # Salvar os parâmetros em um arquivo JSON
        save_model_params(model_params)
        print("Parâmetros do modelo salvos em 'model_params.json'.")

        # Carregar os parâmetros para confirmar
        load_model_params()
        print(f"Parâmetros do modelo carregados: {model_params}")

        return jsonify({
            "message": "Modelo treinado com sucesso!",
            "model_parameters": model_params
        })
    except Exception as e:
        print(f"Erro ao treinar o modelo: {e}")
        return jsonify({"error": f"Erro ao treinar o modelo: {str(e)}"}), 500

# Rota para predições
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model_params  # Declarar que vamos usar a variável global model_params

    if model_params is None:
        print("Parâmetros do modelo não carregados, tentando carregar.")
        load_model_params()
    else:
        print("Parâmetros do modelo já carregados.")

    if model_params is None:
        print("Parâmetros do modelo ainda não estão disponíveis após tentativa de carregamento.")
    else:
        print(f"Parâmetros do modelo disponíveis: {model_params}")

    if request.method == 'POST':
        selected_column = request.form.get('selected_column')
        selected_date = request.form.get('selected_date')
        print(f"Coluna selecionada: {selected_column}")
        print(f"Data selecionada: {selected_date}")

        # Inicializar variáveis para os resultados
        analysis_results = {}
        prediction_result = None

        upload_folder = app.config['UPLOAD_FOLDER']

        # [Código existente para análise da coluna selecionada]

        # Fazer previsão com base na data selecionada
        if selected_date:
            try:
                print(f"Data selecionada para previsão: {selected_date}")
                date_obj = pd.to_datetime(selected_date, errors='coerce', infer_datetime_format=True)
                if pd.isnull(date_obj):
                    raise ValueError("Formato de data inválido. Use YYYY-MM-DD ou DD/MM/YYYY.")
                date_ordinal = date_obj.toordinal()
                print(f"Data ordinal: {date_ordinal}")

                if model_params is None:
                    print("Parâmetros do modelo não estão disponíveis.")
                    raise ValueError("Parâmetros do modelo não carregados.")
                else:
                    print("Parâmetros do modelo disponíveis para previsão.")

                # Recriar a previsão usando os parâmetros
                coef = model_params['coef']
                intercept = model_params['intercept']
                prediction = coef * date_ordinal + intercept
                print(f"Resultado da previsão: {prediction}")

                prediction_result = {
                    'date': selected_date,
                    'prediction': prediction
                }
            except Exception as e:
                print(f"Erro durante a previsão: {e}")
                prediction_result = {'error': f"Erro ao fazer a previsão: {str(e)}"}

        # Renderizar os resultados na template
        return render_template('predict_results.html',
                               selected_column=selected_column,
                               analysis_results=analysis_results,
                               prediction_result=prediction_result)
    else:
        # Em uma requisição GET, coletar colunas únicas e renderizar o formulário
        unique_columns = get_all_unique_columns()
        print("Renderizando o formulário de previsão.")
        return render_template('predict.html', columns=unique_columns)

if __name__ == '__main__':
    app.run(debug=True)