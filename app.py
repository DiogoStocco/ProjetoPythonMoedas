from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Configurações iniciais
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Constantes
MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.pkl')
YEARS_TO_PREDICT = [5, 10, 15, 50]

# Variáveis globais
latest_csv_file = None
model = None

def clean_and_validate_data(file_path):
    """
    Limpa e valida os dados do arquivo CSV.
    Retorna um DataFrame limpo e verifica se há ao menos duas colunas numéricas.
    """
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
        raise ValueError("O arquivo precisa ter pelo menos duas colunas numéricas após limpeza.")

    # Preenchendo valores nulos com a média das colunas
    data = data.fillna(data.mean(numeric_only=True))

    # Verificar se ainda existem NaNs
    if data.isna().any().any():
        raise ValueError("Não foi possível limpar completamente os dados. Verifique o arquivo de entrada.")

    # # Remoção de linhas completamente nulas
    # data = data.dropna(how='all')

    # # Tentar converter todas as colunas para numérico (ignorar erros)
    # for col in data.columns:
    #     data[col] = pd.to_numeric(data[col], errors='coerce')

    # # Remover colunas que ficaram completamente nulas
    # data = data.dropna(axis=1, how='all')

    # # Verificar se há ao menos duas colunas numéricas
    # if data.shape[1] < 2:
    #     raise ValueError("O arquivo precisa ter pelo menos duas colunas numéricas após limpeza")

    # # Preenchendo valores nulos com a média e convertendo para inteiro
    # data = data.fillna(data.mean()).astype(int)

    # Verificar se há ao menos duas colunas numéricas
    # if data.select_dtypes(include=['number']).shape[1] < 2:
    #     raise ValueError("O arquivo precisa ter pelo menos duas colunas numéricas após limpeza")

    return data

def train_model(file_path):
    global model

    data = clean_and_validate_data(file_path)

    # Separação de features e target
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]

    # Treinamento do modelo
    model = LinearRegression()
    model.fit(features, target)

    # Salvando o modelo treinado
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo treinado e salvo em '{MODEL_PATH}'.")

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
            fig = px.line(data, x=column_x, y=column_y, title=f'Relação entre {column_x} e {column_y}')
            graph_html = fig.to_html(full_html=False)
            return render_template('visualize.html', data=data.head(), graph=graph_html)

    return render_template('visualize.html', data=data.head(), graph=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, latest_csv_file

    if model is None:
        return render_template('predict.html', columns=[], predictions=None, error="Modelo ainda não treinado")

    if not latest_csv_file:
        return render_template('predict.html', columns=[], predictions=None, error="Nenhum arquivo CSV disponível")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_csv_file)
    try:
        data = clean_and_validate_data(file_path)
    except ValueError as e:
        return render_template('predict.html', columns=[], predictions=None, error=str(e))

    if request.method == 'POST':
        target_column = request.form.get('target_column')

        if not target_column or target_column not in data.columns:
            return render_template(
                'predict.html',
                columns=data.columns.tolist(),
                predictions=None,
                error="Coluna-alvo inválida"
            )

        features = data.drop(columns=[target_column])
        predictions = {}

        try:
            for year in YEARS_TO_PREDICT:
                input_features = features.mean(axis=0) * year
                prediction = model.predict([input_features])[0]
                predictions[f"{year} anos"] = f"{int(round(prediction)):,}"  # Format as a readable number

            return render_template(
                'predict.html',
                columns=data.columns.tolist(),
                predictions=predictions,
                error=None
            )
        except Exception as e:
            return render_template(
                'predict.html',
                columns=data.columns.tolist(),
                predictions=None,
                error=f"Erro ao realizar a previsão: {str(e)}"
            )

    # Render the initial page with column selection
    return render_template(
        'predict.html',
        columns=data.columns.tolist(),
        predictions=None,
        error=None
    )

if __name__ == '__main__':
    app.run(debug=True)
