import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
import plotly.express as px

# Cargar modelo
with open("modelo_iris.pkl", "rb") as f:
    model, metrics = pickle.load(f)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

st.title("🌸 Iris Species Classification Dashboard")
st.write("Proyecto final - Data Mining")

# ----- Métricas -----
st.header("📊 Métricas del Modelo")
st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
st.write(f"**Precision:** {metrics['precision']:.4f}")
st.write(f"**Recall:** {metrics['recall']:.4f}")
st.write(f"**F1 Score:** {metrics['f1']:.4f}")

# ----- Gráficos -----
st.header("📈 Visualización de datos")

fig = px.scatter_3d(
    df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    z="petal length (cm)",
    color=df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"}),
    title="Iris Dataset en 3D"
)
st.plotly_chart(fig)

# ----- Predicción -----
st.header("🔍 Predicción de especie")

sl = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.0)
sw = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
pl = st.number_input("Petal length (cm)", 0.0, 10.0, 1.5)
pw = st.number_input("Petal width (cm)", 0.0, 10.0, 0.5)

if st.button("Predecir"):
    x_new = np.array([[sl, sw, pl, pw]])
    pred = model.predict(x_new)[0]
    name = iris.target_names[pred]

    st.success(f"🌼 La flor es: **{name.capitalize()}**")

    df2 = df.copy()
    df2["color"] = df2["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    new_point = pd.DataFrame({
        "sepal length (cm)": [sl],
        "sepal width (cm)": [sw],
        "petal length (cm)": [pl],
        "petal length (cm)": [pl],
        "species": ["Nuevo valor"]
    })

    fig2 = px.scatter_3d(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        z="petal length (cm)",
        color=df["species"],
        title="Nueva muestra en el scatter 3D"
    )

    fig2.add_scatter3d(
        x=[sl], y=[sw], z=[pl],
        mode="markers",
        marker=dict(size=8),
        name="Nuevo dato"
    )

    st.plotly_chart(fig2)
