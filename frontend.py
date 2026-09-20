"""Interfaz Streamlit para entrenar un modelo de precios de viviendas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from backend import FEATURES, predict_price, train_housing_model


def demo_housing() -> pd.DataFrame:
    rng = np.random.default_rng(25)
    rows = 180
    area = rng.uniform(40, 240, rows)
    bedrooms = np.clip(np.round(area / 45 + rng.normal(0, 0.8, rows)), 1, 6)
    age = rng.uniform(0, 50, rows)
    price = 45_000 + area * 2_100 + bedrooms * 12_000 - age * 850 + rng.normal(0, 22_000, rows)
    price = np.maximum(price, 30_000)
    return pd.DataFrame({"area_m2": area, "habitaciones": bedrooms, "edad": age, "precio": price})


def read_housing_csv(file) -> pd.DataFrame:
    data = pd.read_csv(file)
    data.columns = [str(column).strip().lower() for column in data.columns]
    required = set(FEATURES + ["precio"])
    if not required.issubset(data.columns):
        raise ValueError("El CSV debe incluir area_m2, habitaciones, edad y precio.")
    data = data[FEATURES + ["precio"]].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        raise ValueError("El CSV contiene datos vacíos o no numéricos.")
    if len(data) < 10:
        raise ValueError("Se requieren al menos 10 viviendas para entrenar y evaluar el modelo.")
    return data


st.set_page_config(page_title="Precio de viviendas", page_icon="🏠", layout="wide")
st.title("Predicción del precio de viviendas")
st.write("Entrena una regresión lineal con datos de viviendas y estima un precio a partir de sus características.")

with st.sidebar:
    source = st.radio("Datos", ["Ejemplo sintético", "Subir CSV"])
    uploaded = st.file_uploader("CSV de viviendas", type=["csv"]) if source == "Subir CSV" else None
if source == "Subir CSV" and uploaded is None:
    st.info("Sube un CSV con las cuatro columnas requeridas.")
    st.stop()
try:
    data = demo_housing() if source == "Ejemplo sintético" else read_housing_csv(uploaded)
    result = train_housing_model(data[FEATURES].to_numpy(), data["precio"].to_numpy())
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

metrics = st.columns(3)
metrics[0].metric("MAE de evaluación", f"{result.mae:,.0f}")
metrics[1].metric("RMSE de evaluación", f"{result.rmse:,.0f}")
metrics[2].metric("R² de evaluación", f"{result.r2:.3f}")
st.caption("La evaluación usa una partición aleatoria de datos. El precio se expresa en las mismas unidades que el CSV.")

st.subheader("Estimar una vivienda")
cols = st.columns(3)
area = cols[0].number_input("Área (m²)", min_value=0.0, value=100.0, step=5.0)
bedrooms = cols[1].number_input("Habitaciones", min_value=0.0, value=3.0, step=1.0)
age = cols[2].number_input("Antigüedad (años)", min_value=0.0, value=10.0, step=1.0)
estimate = predict_price(result.model, area, bedrooms, age)
st.success(f"Precio estimado por el modelo: **{estimate:,.2f}**")

coefficient_table = pd.DataFrame({"Variable": FEATURES, "Coeficiente": result.coefficients})
with st.expander("Coeficientes y datos usados"):
    st.dataframe(coefficient_table, use_container_width=True, hide_index=True)
    st.dataframe(data, use_container_width=True, hide_index=True)
