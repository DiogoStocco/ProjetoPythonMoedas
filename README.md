# **WebApp de Análise de Dados e Machine Learning**

Este projeto é uma aplicação web interativa que permite ao usuário fazer upload de arquivos `.csv`, realizar análises visuais e executar tarefas de machine learning, como treinamento e predição, utilizando diferentes classificadores. 

## **Funcionalidades**

1. **Upload de Dados**  
   - Os usuários podem carregar arquivos `.csv` com dados estruturados.  
   - A aplicação exibe os dados carregados para análise inicial.

2. **Visualização de Dados**  
   - Geração de gráficos interativos com `Plotly` para análise de variáveis.  
   - Gráficos de barras, pizza, dispersão, e histogramas estão disponíveis.  

3. **Treinamento de Modelos de Machine Learning**  
   - Escolha entre modelos de **Regressão Linear** e **K-Nearest Neighbors (KNN)**.  
   - Ajuste de parâmetros do modelo diretamente pela interface (e.g., número de vizinhos no KNN).  
   - O modelo treinado é avaliado e salvo automaticamente utilizando `joblib`.

4. **Predição Dinâmica**  
   - Após o treinamento, os modelos salvos podem ser carregados para realizar predições.  
   - O usuário fornece dados de entrada via JSON, e as predições são retornadas.

5. **Armazenamento de Modelos**  
   - Modelos treinados são salvos no diretório `models` com um timestamp no nome para organização.

---

## **Instalação e Execução**

### **Pré-requisitos**
- Python 3.8+
- Bibliotecas necessárias (listadas em `requirements.txt`).

### **Passo a Passo**

1. **Clone o repositório**
   ```bash
   git clone https://github.com/DiogoStocco/ProjetoPythonMoedas.git
   cd ProjetoPythonMoedas
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure o diretório de uploads**
   Crie as pastas necessárias para armazenar os arquivos de dados e modelos:
   ```bash
   mkdir uploads models
   ```

4. **Execute a aplicação**
   ```bash
   python app.py
   ```
   Acesse a aplicação em: [http://localhost:5000](http://localhost:5000).

---

## **Fluxo da Aplicação**

### 1. **Upload de Dados**
   - Acesse a página inicial.  
   - Faça upload de um arquivo `.csv` e visualize os dados carregados.  

### 2. **Exploração de Dados**
   - Use a seção de gráficos para gerar visualizações interativas com base nas colunas do dataset.  

### 3. **Treinamento de Modelos**
   - Vá até a página de treinamento (`/train/<filename>`).  
   - Escolha as variáveis independentes (X) e a variável dependente (y).  
   - Configure o tipo de modelo (e.g., Regressão Linear ou KNN).  
   - O modelo será treinado e salvo automaticamente.  

### 4. **Predição**
   - Use a rota `/predict` para realizar predições:
     - Forneça o nome do modelo salvo e os dados de entrada em JSON.
     - Receba as predições diretamente na resposta.

---

## **Estrutura do Projeto**

```plaintext
.
├── app.py                 # Código principal da aplicação
├── templates/             # Arquivos HTML para renderização
│   ├── index.html         # Página inicial
│   ├── train.html         # Página de treinamento
├── uploads/               # Diretório para arquivos CSV enviados
├── models/                # Diretório para modelos treinados (salvos como .joblib)
├── static/                # Arquivos estáticos (CSS, JS, etc.)
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

---

## **Endpoints**

### **1. Página Inicial**
- **URL**: `/`
- **Método**: GET
- **Descrição**: Interface para upload de arquivos e visualização inicial.

### **2. Treinamento de Modelos**
- **URL**: `/train/<filename>`
- **Métodos**: GET, POST
- **Descrição**: Treinamento de modelos de machine learning com base nos dados enviados.

### **3. Predição**
- **URL**: `/predict`
- **Método**: POST
- **Formato de Entrada**:
  ```json
  {
    "model_filename": "nome_do_modelo.joblib",
    "input_data": {"coluna1": valor1, "coluna2": valor2}
  }
  ```
- **Formato de Saída**:
  ```json
  {
    "predictions": [resultado1, resultado2]
  }
  ```

---

## **Bibliotecas Utilizadas**

- **Flask**: Framework web para criação da aplicação.
- **Pandas**: Manipulação de dados.
- **Plotly**: Geração de gráficos interativos.
- **Scikit-learn**: Modelos de machine learning.
- **Joblib**: Salvamento e carregamento de modelos treinados.

---

## **Melhorias Futuras**

- Adicionar suporte a mais modelos de machine learning, como árvores de decisão e regressão logística.
- Implementar autenticação de usuários para gerenciar uploads e modelos salvos.
- Integrar um banco de dados para armazenar histórico de uploads e configurações de modelos.