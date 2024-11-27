import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# carregar o dataset Iris
df = sns.load_dataset('iris')

# exibir informações iniciais do Streamlit
st.title("Análise do Conjunto de Dados Iris")
st.write(df.head())

# Estatísticas descritivas
st.subheader("Estatísticas Descritivas")
st.write(df.describe())

# Gráficos interativos
st.subheader("Gráfico de Dispersão: Sepal Length vs Sepal Width")
st.write("Visualização das características das espécies de iris.")
scatter_plot = sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
st.pyplot(scatter_plot.figure)

# Gráfico de histograma
st.subheader("Distribuição do Comprimento da Pétala")
st.write("Distribuição do comprimento da pétala para as três espécies.")
hist_plot = sns.histplot(data=df, x='petal_length', hue='species', multiple='stack')
st.pyplot(hist_plot.figure)
