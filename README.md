# 🌸 Aplicativo Streamlit - Análise do Dataset Iris

Este projeto apresenta uma análise completa do famoso dataset Iris usando machine learning, implementada como um aplicativo web interativo com Streamlit.

## 📋 Descrição do Projeto

O aplicativo oferece uma documentação completa e interativa da análise do dataset Iris, incluindo:

- **🏠 Introdução**: Descrição do problema, dataset e objetivos
- **📊 Análise Exploratória de Dados (EDA)**: Visualizações e estatísticas descritivas
- **🔧 Metodologia**: Detalhamento dos passos de pré-processamento e escolha do modelo
- **📈 Resultados**: Métricas de avaliação e matriz de confusão
- **💡 Conclusão**: Resumo das descobertas e limitações
- **📚 Referências**: Links para datasets e fontes relevantes

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone ou baixe os arquivos do projeto**
   ```bash
   # Se usando git
   git clone <url-do-repositorio>
   cd iris-streamlit-app
   
   # Ou simplesmente baixe os arquivos:
   # - streamlit_app.py
   # - iris.csv
   # - requirements.txt
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

### Executando o Aplicativo

1. **Execute o comando Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Acesse o aplicativo**
   - O aplicativo será aberto automaticamente no seu navegador
   - Ou acesse manualmente: `http://localhost:8501`

## 📁 Estrutura dos Arquivos

```
iris-streamlit-app/
├── streamlit_app.py      # Aplicativo principal Streamlit
├── iris.csv             # Dataset Iris
├── requirements.txt     # Dependências Python
├── README.md           # Este arquivo
└── screenshots/        # Screenshots do aplicativo (opcional)
```

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para criação de aplicativos web
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Matplotlib & Seaborn**: Visualização de dados estática
- **Plotly**: Visualização de dados interativa
- **Scikit-learn**: Machine learning e métricas de avaliação

## 📊 Funcionalidades

### Navegação Interativa
- Menu lateral com navegação entre seções
- Interface responsiva e intuitiva

### Análise Exploratória
- Estatísticas descritivas detalhadas
- Visualizações interativas com Plotly
- Matriz de correlação
- Box plots e scatter plots

### Machine Learning
- Implementação de Decision Tree Classifier
- Métricas de avaliação completas
- Matriz de confusão interativa
- Análise de importância das características

### Documentação Completa
- Metodologia detalhada
- Interpretação dos resultados
- Limitações e sugestões de melhorias
- Referências e links úteis

## 🎯 Resultados do Modelo

O modelo Decision Tree alcançou excelente performance:

- **Acurácia**: 97.4%
- **Precisão**: 97.6%
- **Recall**: 97.4%
- **F1-Score**: 97.4%

## 📈 Principais Descobertas

1. **Dataset Balanceado**: 50 amostras de cada espécie
2. **Qualidade dos Dados**: Sem valores nulos ou duplicatas
3. **Separabilidade**: Iris Setosa é facilmente distinguível
4. **Características Importantes**: Pétalas são mais discriminativas que sépalas

## 🔧 Personalização

O aplicativo pode ser facilmente personalizado:

- **Cores e Estilo**: Modifique o CSS no início do arquivo
- **Visualizações**: Adicione novos gráficos ou modifique os existentes
- **Modelos**: Experimente diferentes algoritmos de machine learning
- **Dados**: Substitua por outros datasets similares

## 📝 Estrutura do Código

```python
# Principais seções do streamlit_app.py:

1. Configuração e Imports
2. Funções de Cache (@st.cache_data)
3. Interface Principal (main())
4. Seções Navegáveis:
   - Introdução
   - EDA
   - Metodologia
   - Resultados
   - Conclusão
   - Referências
```

## 🐛 Solução de Problemas

### Erro de Importação
```bash
# Se houver erro com alguma biblioteca:
pip install --upgrade <nome-da-biblioteca>
```

### Porta em Uso
```bash
# Se a porta 8501 estiver ocupada:
streamlit run streamlit_app.py --server.port 8502
```

### Dataset Não Encontrado
- Certifique-se de que o arquivo `iris.csv` está no mesmo diretório
- Ou modifique o caminho no código: `pd.read_csv('caminho/para/iris.csv')`

## 📚 Recursos Adicionais

- [Documentação do Streamlit](https://docs.streamlit.io/)
- [Dataset Iris no Kaggle](https://www.kaggle.com/datasets/uciml/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Plotly Python Documentation](https://plotly.com/python/)

## 🤝 Contribuições

Sugestões de melhorias são bem-vindas:

1. **Novos Modelos**: Random Forest, SVM, Neural Networks
2. **Visualizações**: Gráficos 3D, animações
3. **Funcionalidades**: Upload de dados, comparação de modelos
4. **Interface**: Melhorias de UX/UI

## 📄 Licença

Este projeto é para fins educacionais e pode ser usado livremente para aprendizado e demonstração.

## 👨‍💻 Autor

Desenvolvido como demonstração de análise de dados e machine learning com Streamlit.

---

**🌸 Aproveite explorando o mundo fascinante da análise de dados com o dataset Iris! 🌸**

