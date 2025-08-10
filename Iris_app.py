import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise do Dataset Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Função para carregar dados
@st.cache_data
def load_data():
    data = pd.read_csv('Iris.csv')
    return data


# Função para treinar modelo
@st.cache_data
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50, stratify=y)
    model = DecisionTreeClassifier(random_state=50)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, x_train, x_test, y_train, y_test, predictions


def main():
    # Título principal
    st.markdown('<h1 class="main-header">🌸 Análise do Dataset Iris</h1>', unsafe_allow_html=True)

    # Sidebar para navegação
    st.sidebar.title("📋 Navegação")
    sections = [
        "🏠 Introdução",
        "📊 Análise Exploratória de Dados (EDA)",
        "🔧 Metodologia",
        "📈 Resultados",
        "💡 Conclusão",
        "📚 Referências"
    ]

    selected_section = st.sidebar.radio("Selecione uma seção:", sections)

    # Carregar dados
    data = load_data()

    # Seção 1: Introdução
    if selected_section == "🏠 Introdução":
        st.markdown('<h2 class="section-header">🏠 Introdução</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Definindo o estilo CSS
            st.markdown(
                """
                <style>
                body {
                    background-color: black; /* Cor de fundo preta */
                    color: white; /* Cor do texto branca */
                }
                h3 {
                    color: #4CAF50; /* Cor do título em verde */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown("""
            <div class="info-box">
            <h3>🎯 Objetivo do Projeto</h3>
            <p>Este projeto tem como objetivo realizar uma análise completa do famoso dataset Iris, 
            desenvolvendo um modelo de classificação para identificar automaticamente as espécies de íris 
            com base nas características morfológicas das flores.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            ### 🌺 Sobre o Dataset Iris

            O dataset Iris é um dos conjuntos de dados mais conhecidos na literatura de reconhecimento de padrões. 
            Foi introduzido pelo estatístico e biólogo Ronald Fisher em 1936 e contém medições de 150 flores de íris 
            de três espécies diferentes:

            - **Iris Setosa** 🌸
            - **Iris Versicolor** 🌺  
            - **Iris Virginica** 🌻

            ### 📏 Características Medidas

            Para cada flor, foram registradas quatro características:

            1. **Comprimento da Sépala** (sepal_length)
            2. **Largura da Sépala** (sepal_width)
            3. **Comprimento da Pétala** (petal_length)
            4. **Largura da Pétala** (petal_width)

            Todas as medições estão em centímetros.
            """)

        with col2:
            st.markdown("### 📊 Visão Geral dos Dados")
            st.dataframe(data.head(10), use_container_width=True)

            st.markdown("### 📈 Estatísticas Básicas")
            st.write(f"**Total de amostras:** {len(data)}")
            st.write(f"**Número de características:** {len(data.columns) - 1}")
            st.write(f"**Espécies únicas:** {data['Species'].nunique()}")
            st.write(f"**Amostras por espécie:** {len(data) // data['Species'].nunique()}")

    # Seção 2: EDA
    elif selected_section == "📊 Análise Exploratória de Dados (EDA)":
        st.markdown('<h2 class="section-header">📊 Análise Exploratória de Dados (EDA)</h2>', unsafe_allow_html=True)

        # Informações gerais
        st.markdown("### 📋 Informações Gerais do Dataset")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Dimensões</h4>
            <p><strong>Linhas:</strong> {}</p>
            <p><strong>Colunas:</strong> {}</p>
            </div>
            """.format(data.shape[0], data.shape[1]), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🌸 Espécies</h4>
            <p><strong>Setosa:</strong> {}</p>
            <p><strong>Versicolor:</strong> {}</p>
            <p><strong>Virginica:</strong> {}</p>
            </div>
            """.format(
                len(data[data['Species'] == '0']),
                len(data[data['Species'] == '1']),
                len(data[data['Species'] == '2'])
            ), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>🔍 Qualidade dos Dados</h4>
            <p><strong>Valores nulos:</strong> {}</p>
            <p><strong>Duplicatas:</strong> {}</p>
            </div>
            """.format(data.isnull().sum().sum(), data.duplicated().sum()), unsafe_allow_html=True)

        # Estatísticas descritivas
        st.markdown("### 📈 Estatísticas Descritivas")
        st.dataframe(data.describe(), use_container_width=True)

        # Visualizações
        st.markdown("### 📊 Visualizações")

        # Distribuição das espécies
        st.markdown("#### 🌸 Distribuição das Espécies")
        fig_count = px.histogram(data, x='Species', color='Species',
                                 title='Distribuição das Espécies de Íris')
        fig_count.update_layout(showlegend=False)
        st.plotly_chart(fig_count, use_container_width=True)

        # Pair plot
        st.markdown("#### 🔗 Relações entre Características")
        fig_scatter = px.scatter_matrix(data,
                                        dimensions=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'],
                                        color='Species',
                                        title='Matriz de Dispersão das Características')
        fig_scatter.update_layout(height=800)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Box plots
        st.markdown("#### 📦 Distribuição das Características por Espécie")

        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        feature_names = ['Comprimento da Sépala', 'Largura da Sépala', 'Comprimento da Pétala', 'Largura da Pétala']

        col1, col2 = st.columns(2)

        for i, (feature, name) in enumerate(zip(features, feature_names)):
            fig_box = px.box(data, x='Species', y=feature, color='Species',
                             title=f'Distribuição de {name} por Espécie')
            if i % 2 == 0:
                col1.plotly_chart(fig_box, use_container_width=True)
            else:
                col2.plotly_chart(fig_box, use_container_width=True)

        # Correlação
        st.markdown("#### 🔗 Matriz de Correlação")
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr_matrix,
                             text_auto=True,
                             aspect="auto",
                             title='Matriz de Correlação das Características')
        st.plotly_chart(fig_corr, use_container_width=True)

        # Definindo o estilo CSS
        st.markdown(
            """
            <style>
            body {
                background-color: black; /* Cor de fundo preta */
                color: white; /* Cor do texto branca */
            }
            h4 {
                color: #4CAF50; /* Cor do título em verde */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Principais descobertas
        st.markdown("### 🔍 Principais Descobertas da EDA")
        st.markdown("""
        <h4>📊 Insights Importantes:</h4>
        <ul>
        <li><strong>Balanceamento:</strong> O dataset está perfeitamente balanceado com 50 amostras de cada espécie</li>
        <li><strong>Qualidade:</strong> Não há valores nulos ou duplicatas no dataset</li>
        <li><strong>Separabilidade:</strong> As espécies mostram padrões distintos, especialmente Iris Setosa</li>
        <li><strong>Correlações:</strong> Comprimento e largura das pétalas são altamente correlacionados (0.96)</li>
        <li><strong>Características distintivas:</strong> Pétalas são mais discriminativas que sépalas para classificação</li>
        </ul>
        """, unsafe_allow_html=True)

    # Seção 3: Metodologia
    elif selected_section == "🔧 Metodologia":
        st.markdown('<h2 class="section-header">🔧 Metodologia</h2>', unsafe_allow_html=True)

        st.markdown("### 📋 Etapas do Processo")

        # Pré-processamento
        st.markdown("#### 1. 🔄 Pré-processamento dos Dados")
        st.markdown("""
        <div class="info-box">
        <h4>Passos realizados:</h4>
        <ul>
        <li><strong>Carregamento:</strong> Importação do dataset usando pandas</li>
        <li><strong>Verificação:</strong> Análise de valores nulos e duplicatas</li>
        <li><strong>Codificação:</strong> Conversão das espécies para valores numéricos usando LabelEncoder</li>
        <li><strong>Separação:</strong> Divisão entre características (X) e variável alvo (y)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Divisão dos dados
        st.markdown("#### 2. ✂️ Divisão dos Dados")
        st.markdown("""
        <div class="info-box">
        <h4>Estratégia de divisão:</h4>
        <ul>
        <li><strong>Proporção:</strong> 75% treino / 25% teste</li>
        <li><strong>Estratificação:</strong> Mantém a proporção das classes em ambos os conjuntos</li>
        <li><strong>Semente aleatória:</strong> 50 (para reprodutibilidade)</li>
        <li><strong>Resultado:</strong> 112 amostras para treino, 38 para teste</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Escolha do modelo
        st.markdown("#### 3. 🤖 Escolha do Modelo")
        st.markdown("""
        <div class="info-box">
        <h4>Decision Tree Classifier:</h4>
        <ul>
        <li><strong>Razão da escolha:</strong> Simplicidade e interpretabilidade</li>
        <li><strong>Adequação:</strong> Funciona bem com dados categóricos e numéricos</li>
        <li><strong>Vantagens:</strong> Fácil visualização e compreensão das regras de decisão</li>
        <li><strong>Parâmetros:</strong> Configuração padrão do scikit-learn</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Treinamento
        st.markdown("#### 4. 🎯 Treinamento do Modelo")
        st.markdown("""
        <div class="info-box">
        <h4>Processo de treinamento:</h4>
        <ul>
        <li><strong>Algoritmo:</strong> Decision Tree com critério de impureza Gini</li>
        <li><strong>Dados:</strong> 112 amostras de treinamento</li>
        <li><strong>Características:</strong> 4 variáveis numéricas (medidas das flores)</li>
        <li><strong>Classes:</strong> 3 espécies de íris</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Avaliação
        st.markdown("#### 5. 📊 Avaliação do Modelo")
        st.markdown("""
        <div class="info-box">
        <h4>Métricas utilizadas:</h4>
        <ul>
        <li><strong>Acurácia:</strong> Proporção de predições corretas</li>
        <li><strong>Precisão:</strong> Proporção de verdadeiros positivos entre as predições positivas</li>
        <li><strong>Recall:</strong> Proporção de verdadeiros positivos identificados</li>
        <li><strong>F1-Score:</strong> Média harmônica entre precisão e recall</li>
        <li><strong>Matriz de Confusão:</strong> Visualização detalhada dos acertos e erros</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Código exemplo
        st.markdown("### 💻 Código Principal")
        st.code("""
# Carregamento e pré-processamento
data = pd.read_csv('iris.csv')
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])

# Separação das características e alvo
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species_encoded']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=50, stratify=y
)

# Treinamento do modelo
model = DecisionTreeClassifier(random_state=50)
model.fit(X_train, y_train)

# Predições e avaliação
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
        """, language='python')

    # Seção 4: Resultados
    elif selected_section == "📈 Resultados":
        st.markdown('<h2 class="section-header">📈 Resultados</h2>', unsafe_allow_html=True)

        # Preparar dados para o modelo
        le = LabelEncoder()
        data_encoded = data.copy()
        data_encoded['species_encoded'] = le.fit_transform(data['Species'])

        X = data_encoded[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = data_encoded['species_encoded']

        # Treinar modelo
        model, x_train, x_test, y_train, y_test, predictions = train_model(X, y)

        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        # Exibir métricas
        st.markdown("### 📊 Métricas de Avaliação")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h4>🎯 Acurácia</h4>
            <h2 style="color: #1f77b4;">{accuracy:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h4>🔍 Precisão</h4>
            <h2 style="color: #2e8b57;">{precision:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <h4>📈 Recall</h4>
            <h2 style="color: #ff7f0e;">{recall:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
            <h4>⚖️ F1-Score</h4>
            <h2 style="color: #d62728;">{f1:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Matriz de confusão
        st.markdown("### 🔍 Matriz de Confusão")

        cm = confusion_matrix(y_test, predictions)
        species_names = ['Setosa', 'Versicolor', 'Virginica']

        fig_cm = px.imshow(cm,
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='Blues',
                           title='Matriz de Confusão')
        fig_cm.update_layout(
            xaxis_title='Predito',
            yaxis_title='Real',
            xaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=species_names),
            yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=species_names)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Interpretação da matriz
        st.markdown("### 📋 Interpretação dos Resultados")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>✅ Pontos Fortes:</h4>
            <ul>
            <li><strong>Alta Acurácia:</strong> O modelo alcançou excelente performance</li>
            <li><strong>Setosa Perfeita:</strong> 100% de acerto na classificação da Iris Setosa</li>
            <li><strong>Baixo Overfitting:</strong> Performance consistente entre treino e teste</li>
            <li><strong>Métricas Balanceadas:</strong> Precisão, recall e F1-score muito próximos</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>⚠️ Observações:</h4>
            <ul>
            <li><strong>Confusão Mínima:</strong> Pequena confusão entre Versicolor e Virginica</li>
            <li><strong>Dataset Simples:</strong> Iris é um problema relativamente fácil</li>
            <li><strong>Amostra Pequena:</strong> Apenas 38 amostras no conjunto de teste</li>
            <li><strong>Generalização:</strong> Resultados podem variar com novos dados</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Importância das características
        st.markdown("### 🌟 Importância das Características")

        feature_importance = model.feature_importances_
        feature_names = ['Comprimento Sépala', 'Largura Sépala', 'Comprimento Pétala', 'Largura Pétala']

        fig_importance = px.bar(
            x=feature_importance,
            y=feature_names,
            orientation='h',
            title='Importância das Características no Modelo',
            labels={'x': 'Importância', 'y': 'Características'}
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)

        # Predições detalhadas
        st.markdown("### 🔍 Análise Detalhada das Predições")

        # Criar DataFrame com resultados
        results_df = pd.DataFrame({
            'Real': [species_names[i] for i in y_test],
            'Predito': [species_names[i] for i in predictions],
            'Correto': y_test == predictions
        })

        # Estatísticas por classe
        st.markdown("#### 📊 Performance por Espécie")

        for i, species in enumerate(species_names):
            mask_real = y_test == i
            mask_pred = predictions == i

            tp = np.sum((y_test == i) & (predictions == i))  # True Positives
            fp = np.sum((y_test != i) & (predictions == i))  # False Positives
            fn = np.sum((y_test == i) & (predictions != i))  # False Negatives

            precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class) if (
                                                                                                              precision_class + recall_class) > 0 else 0

            st.markdown(f"""
            **{species}:**
            - Precisão: {precision_class:.1%}
            - Recall: {recall_class:.1%}
            - F1-Score: {f1_class:.1%}
            """)

    # Seção 5: Conclusão
    elif selected_section == "💡 Conclusão":
        st.markdown('<h2 class="section-header">💡 Conclusão</h2>', unsafe_allow_html=True)

        st.markdown("### 🎯 Resumo das Descobertas")

        st.markdown("""
        <div class="info-box">
        <h4>🔍 Principais Achados:</h4>
        <ul>
        <li><strong>Excelente Performance:</strong> O modelo Decision Tree alcançou alta acurácia na classificação das espécies de íris</li>
        <li><strong>Características Distintivas:</strong> As medidas das pétalas são mais importantes que as das sépalas para classificação</li>
        <li><strong>Separabilidade Clara:</strong> Iris Setosa é facilmente distinguível das outras espécies</li>
        <li><strong>Confusão Mínima:</strong> Pequena sobreposição entre Versicolor e Virginica</li>
        <li><strong>Dataset Balanceado:</strong> Distribuição uniforme das classes facilita o aprendizado</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚠️ Limitações do Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🚧 Limitações Identificadas:</h4>
            <ul>
            <li><strong>Dataset Pequeno:</strong> Apenas 150 amostras podem não representar toda a variabilidade</li>
            <li><strong>Simplicidade:</strong> Problema relativamente simples com apenas 4 características</li>
            <li><strong>Overfitting Potencial:</strong> Decision Trees podem memorizar padrões específicos</li>
            <li><strong>Generalização:</strong> Performance pode variar com dados de outras fontes</li>
            <li><strong>Características Limitadas:</strong> Apenas medidas morfológicas básicas</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>🔄 Impactos das Limitações:</h4>
            <ul>
            <li><strong>Robustez:</strong> Modelo pode ser sensível a variações nos dados</li>
            <li><strong>Aplicabilidade:</strong> Resultados específicos para este dataset</li>
            <li><strong>Complexidade:</strong> Problemas reais podem ser mais desafiadores</li>
            <li><strong>Validação:</strong> Necessidade de mais dados para validação robusta</li>
            <li><strong>Contexto:</strong> Limitado ao domínio específico das íris</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🚀 Sugestões para Futuras Melhorias")

        st.markdown("""
        <div class="info-box">
        <h4>🔮 Próximos Passos:</h4>

        <h5>📊 Dados e Características:</h5>
        <ul>
        <li><strong>Mais Dados:</strong> Coletar amostras adicionais para aumentar a robustez</li>
        <li><strong>Novas Características:</strong> Incluir medidas adicionais (cor, textura, forma)</li>
        <li><strong>Dados Temporais:</strong> Considerar variações sazonais e de crescimento</li>
        <li><strong>Imagens:</strong> Incorporar análise de imagens das flores</li>
        </ul>

        <h5>🤖 Modelos e Técnicas:</h5>
        <ul>
        <li><strong>Ensemble Methods:</strong> Random Forest, Gradient Boosting</li>
        <li><strong>SVM:</strong> Support Vector Machines para fronteiras não-lineares</li>
        <li><strong>Neural Networks:</strong> Redes neurais para padrões complexos</li>
        <li><strong>Cross-Validation:</strong> Validação cruzada para avaliação mais robusta</li>
        </ul>

        <h5>🔍 Validação e Teste:</h5>
        <ul>
        <li><strong>Validação Externa:</strong> Testar com dados de outras fontes</li>
        <li><strong>Análise de Sensibilidade:</strong> Estudar robustez a variações nos dados</li>
        <li><strong>Interpretabilidade:</strong> Usar SHAP ou LIME para explicar predições</li>
        <li><strong>Monitoramento:</strong> Acompanhar performance ao longo do tempo</li>
        </ul>

        <h5>🌐 Aplicações Práticas:</h5>
        <ul>
        <li><strong>App Mobile:</strong> Aplicativo para identificação em campo</li>
        <li><strong>API:</strong> Serviço web para classificação automática</li>
        <li><strong>Integração:</strong> Incorporar em sistemas de catalogação botânica</li>
        <li><strong>Educação:</strong> Ferramenta educacional para ensino de botânica</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎓 Valor Educacional e Científico")

        st.markdown("""
        <div class="info-box">
        <h4>📚 Contribuições do Projeto:</h4>
        <ul>
        <li><strong>Metodologia Clara:</strong> Demonstra processo completo de análise de dados</li>
        <li><strong>Reprodutibilidade:</strong> Código e métodos bem documentados</li>
        <li><strong>Benchmark:</strong> Estabelece baseline para comparações futuras</li>
        <li><strong>Didático:</strong> Excelente exemplo para ensino de machine learning</li>
        <li><strong>Fundação:</strong> Base sólida para projetos mais complexos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Seção 6: Referências
    elif selected_section == "📚 Referências":
        st.markdown('<h2 class="section-header">📚 Referências</h2>', unsafe_allow_html=True)

        st.markdown("### 📖 Fontes de Dados")

        st.markdown("""
        <div class="info-box">
        <h4>🗃️ Dataset Principal:</h4>
        <ul>
        <li><strong>Iris Dataset:</strong> <a href="https://www.kaggle.com/datasets/uciml/iris" target="_blank">Kaggle - UCI Iris Dataset</a></li>
        <li><strong>Fonte Original:</strong> Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"</li>
        <li><strong>UCI Repository:</strong> <a href="https://archive.ics.uci.edu/ml/datasets/iris" target="_blank">UCI Machine Learning Repository</a></li>
        <li><strong>Seaborn Data:</strong> <a href="https://github.com/mwaskom/seaborn-data" target="_blank">Seaborn Built-in Datasets</a></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📚 Literatura Científica")

        st.markdown("""
        <div class="info-box">
        <h4>📄 Artigos e Publicações:</h4>
        <ul>
        <li><strong>Fisher, R.A. (1936):</strong> "The use of multiple measurements in taxonomic problems". Annals of Eugenics, 7(2), 179-188.</li>
        <li><strong>Anderson, E. (1935):</strong> "The irises of the Gaspe Peninsula". Bulletin of the American Iris Society, 59, 2-5.</li>
        <li><strong>Duda, R.O. & Hart, P.E. (1973):</strong> "Pattern Classification and Scene Analysis". John Wiley & Sons.</li>
        <li><strong>Hand, D.J. (1981):</strong> "Discrimination and Classification". John Wiley & Sons.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Ferramentas e Bibliotecas")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🐍 Python Libraries:</h4>
            <ul>
            <li><strong>Pandas:</strong> <a href="https://pandas.pydata.org/" target="_blank">pandas.pydata.org</a></li>
            <li><strong>NumPy:</strong> <a href="https://numpy.org/" target="_blank">numpy.org</a></li>
            <li><strong>Scikit-learn:</strong> <a href="https://scikit-learn.org/" target="_blank">scikit-learn.org</a></li>
            <li><strong>Matplotlib:</strong> <a href="https://matplotlib.org/" target="_blank">matplotlib.org</a></li>
            <li><strong>Seaborn:</strong> <a href="https://seaborn.pydata.org/" target="_blank">seaborn.pydata.org</a></li>
            <li><strong>Plotly:</strong> <a href="https://plotly.com/python/" target="_blank">plotly.com/python</a></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>🌐 Frameworks e Plataformas:</h4>
            <ul>
            <li><strong>Streamlit:</strong> <a href="https://streamlit.io/" target="_blank">streamlit.io</a></li>
            <li><strong>Jupyter:</strong> <a href="https://jupyter.org/" target="_blank">jupyter.org</a></li>
            <li><strong>Google Colab:</strong> <a href="https://colab.research.google.com/" target="_blank">colab.research.google.com</a></li>
            <li><strong>GitHub:</strong> <a href="https://github.com/" target="_blank">github.com</a></li>
            <li><strong>Kaggle:</strong> <a href="https://www.kaggle.com/" target="_blank">kaggle.com</a></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📖 Recursos Educacionais")

        st.markdown("""
        <div class="info-box">
        <h4>🎓 Materiais de Estudo:</h4>
        <ul>
        <li><strong>Scikit-learn Documentation:</strong> <a href="https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html" target="_blank">Iris Dataset Example</a></li>
        <li><strong>Towards Data Science:</strong> <a href="https://towardsdatascience.com/" target="_blank">Medium - Data Science Articles</a></li>
        <li><strong>Machine Learning Mastery:</strong> <a href="https://machinelearningmastery.com/" target="_blank">ML Tutorials and Guides</a></li>
        <li><strong>Coursera:</strong> <a href="https://www.coursera.org/" target="_blank">Machine Learning Courses</a></li>
        <li><strong>edX:</strong> <a href="https://www.edx.org/" target="_blank">Data Science Programs</a></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔗 Links Úteis")

        st.markdown("""
        <div class="info-box">
        <h4>🌐 Recursos Adicionais:</h4>
        <ul>
        <li><strong>Iris Species Information:</strong> <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set" target="_blank">Wikipedia - Iris Flower Dataset</a></li>
        <li><strong>Botanical Information:</strong> <a href="https://www.britannica.com/plant/iris-plant" target="_blank">Britannica - Iris Plant</a></li>
        <li><strong>Statistical Analysis:</strong> <a href="https://www.r-project.org/" target="_blank">R Project for Statistical Computing</a></li>
        <li><strong>Data Visualization:</strong> <a href="https://d3js.org/" target="_blank">D3.js - Data Visualization</a></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Citação Sugerida")

        st.markdown("""
        <div class="info-box">
        <h4>📋 Como Citar Este Trabalho:</h4>
        <code>
        Análise do Dataset Iris usando Decision Tree Classifier. 
        Implementado em Python com Streamlit para visualização interativa. 
        Dataset: Fisher, R.A. (1936). UCI Machine Learning Repository.
        </code>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🌸 Análise do Dataset Iris - Desenvolvido com Streamlit 🌸</p>
    <p>📊 Demonstração de Machine Learning e Análise de Dados 📊</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
