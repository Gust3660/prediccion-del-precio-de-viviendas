"""Modelo de regresión para estimar precios de viviendas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ["area_m2", "habitaciones", "edad"]


@dataclass(frozen=True)
class HousingModel:
    model: LinearRegression
    mae: float
    rmse: float
    r2: float
    coefficients: np.ndarray
    intercept: float


def train_housing_model(features: np.ndarray, prices: np.ndarray, test_fraction: float = 0.2) -> HousingModel:
    x = np.asarray(features, dtype=float)
    y = np.asarray(prices, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (x.shape[0],) or x.shape[0] < 10:
        raise ValueError("Se requieren al menos 10 filas con area_m2, habitaciones y edad, además de precio.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x < 0) or np.any(y <= 0):
        raise ValueError("Las características deben ser finitas y no negativas; precio debe ser positivo.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction, random_state=42)
    evaluation_model = LinearRegression().fit(x_train, y_train)
    prediction = evaluation_model.predict(x_test)
    model = LinearRegression().fit(x, y)
    return HousingModel(
        model=model,
        mae=float(mean_absolute_error(y_test, prediction)),
        rmse=float(np.sqrt(mean_squared_error(y_test, prediction))),
        r2=float(r2_score(y_test, prediction)),
        coefficients=np.asarray(model.coef_),
        intercept=float(model.intercept_),
    )


def predict_price(model: LinearRegression, area_m2: float, bedrooms: float, age: float) -> float:
    values = np.asarray([[area_m2, bedrooms, age]], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Las características de la vivienda deben ser números no negativos.")
    return float(model.predict(values)[0])
