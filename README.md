# Predicción del precio de viviendas

Aplicación Streamlit que entrena una regresión lineal para estimar precios de viviendas usando área, número de habitaciones y antigüedad. Incluye un conjunto sintético y admite datos CSV propios.

## Funciones

- Usa datos de ejemplo o carga un archivo propio.
- Entrena una regresión lineal y reporta MAE, RMSE y R² en un conjunto de evaluación.
- Permite introducir las características de una vivienda y consultar una estimación.
- Muestra coeficientes y datos cargados.

## Estructura

```text
.
├── backend.py       # Entrenamiento, métricas y predicción
├── frontend.py      # Interfaz Streamlit
├── requirements.txt
└── README.md
```

## Formato del CSV

Se requieren estas columnas numéricas:

```csv
area_m2,habitaciones,edad,precio
85,2,12,210000
120,3,5,325000
```

Incluye al menos 10 filas; todas las características deben ser no negativas y el precio debe ser positivo. El programa no asume una moneda específica.

## Instalación y ejecución

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run frontend.py
```

En macOS o Linux, activa el entorno con `source .venv/bin/activate`.

## Alcance

El modelo es una regresión lineal simple y no incluye ubicación, condición, tasas de interés ni validación temporal. La estimación es educativa y depende de la calidad y representatividad de los datos proporcionados.
