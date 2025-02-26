import joblib
import pandas as pd
import numpy as np
import os
import logging

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

input_file = './tmp/TEST.csv'
import joblib
import os

# Obtener la ruta correcta dentro del contenedor
model_path = os.path.join(os.path.dirname(__file__), "model/modelRLOG.joblib")


# Cargar los objetos serializados
try:
    model = joblib.load(model_path)
    scaler = joblib.load('model/scaler.joblib')
    knn_imputer = joblib.load('model/knn_imputer.joblib')
    median_values = joblib.load('model/median_values.joblib')
    mode_values = joblib.load('model/mode_values.joblib')
    logging.info("Modelos y transformadores cargados exitosamente.")
except Exception as e:
    logging.error(f"Error cargando modelos y transformadores: {e}")
    raise

# Diccionario de transformación para direcciones de viento
diccionario = {
    'N': ['N', 'NNW', 'NNE', 'NE', 'NW'],
    'S': ['S', 'SSW', 'SSE', 'SE', 'SW'],
    'E': ['E', 'ENE', 'ESE'],
    'W': ['W', 'WNW', 'WSW'],
}
diccionario_invertido = {valor: clave for clave, lista_valores in diccionario.items() for valor in lista_valores}

# Columnas a estandarizar
columns_to_standardize = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                          'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                          'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
                          'Cloud3pm', 'Temp9am', 'Temp3pm', "RainfallTomorrow"]

def preprocesar(data):
    try:
        logging.info("Iniciando preprocesamiento de datos.")
        logging.debug(f"Columnas iniciales en los datos: {list(data.columns)}")

        # Normalizar formato numérico
        for col in data.select_dtypes(include=['object']).columns:
            try:
                data[col] = data[col].str.replace('.', '', regex=True).str.replace(',', '.', regex=True).astype(float)
            except ValueError:
                pass  # Ignorar columnas que no sean numéricas

        # Mapear direcciones de viento
        for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            if col in data.columns:
                data[col] = data[col].map(diccionario_invertido).fillna(data[col])

        # Convertir la columna 'Date' a número de días desde una fecha base
        if 'Date' in data.columns:
            base_date = pd.to_datetime('2000-01-01')
            data['Date'] = pd.to_datetime(data['Date'])
            data['Date'] = (data['Date'] - base_date).dt.days

        # Imputación de valores faltantes
        numeric_cols = list(set(data.select_dtypes(include='number').columns) & set(median_values.keys()))
        categorical_cols = list(set(data.select_dtypes(include='object').columns) & set(mode_values.keys()))

        data[numeric_cols] = data[numeric_cols].fillna(median_values)
        data[categorical_cols] = data[categorical_cols].fillna(mode_values)
        
        # Aplicar imputación solo en las columnas con las que fue entrenado el KNN Imputer
        imputer_columns = knn_imputer.feature_names_in_
        imputer_cols_present = [col for col in imputer_columns if col in data.columns]
        data[imputer_cols_present] = knn_imputer.transform(data[imputer_cols_present])

        # Aplicar One Hot Encoding
        categorical_cols_to_dummy = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        categorical_cols_to_dummy = [col for col in categorical_cols_to_dummy if col in data.columns]
        data = pd.get_dummies(data, columns=categorical_cols_to_dummy, drop_first=True)

        # Validar consistencia con el modelo
        columnas_entrenamiento = model.feature_names_in_
        missing_cols = set(columnas_entrenamiento) - set(data.columns)
        unexpected_cols = set(data.columns) - set(columnas_entrenamiento)

        for col in missing_cols:
            data[col] = 0
        
        
        # Estandarización
        if columns_to_standardize:
            standardize_cols_present = [col for col in columns_to_standardize if col in data.columns]
            data[standardize_cols_present] = scaler.transform(data[standardize_cols_present])

        data = data.drop(columns=unexpected_cols, errors='ignore')
        data = data[columnas_entrenamiento]
        logging.info("Preprocesamiento finalizado correctamente.")
        return data
    except Exception as e:
        # logging.error(f"Error en preprocesamiento: {e}")
        raise

def predecir(data):
    try:
        data_preprocesada = preprocesar(data.copy())
        return model.predict(data_preprocesada)
    except Exception as e:
        # logging.error(f"Error en predicción: {e}")
        raise

# Punto de entrada
if __name__ == '__main__':
    # print("Ingrese la ruta del archivo CSV:")
    # input_file = input("> ")

    logging.info("asd")
    
    try:
        data = pd.read_csv(input_file)
        
        missing_cols = set(model.feature_names_in_) - set(data.columns)
        for col in missing_cols:
            data[col] = 0

        if 'WindSpeed9am' not in data.columns:
            data['WindSpeed9am'] = 0
        
        predictions = predecir(data)
        data['Prediction'] = predictions
        output_file = './tmp/output.csv'
        data.replace({"Prediction": {0: "no", 1: "si"}}).to_csv(output_file, index=False)
        print(f"Predicciones guardadas en {output_file}")
        logging.info(f"Predicciones guardadas en {output_file}")
    # except FileNotFoundError:
        # logging.error("Error: El archivo especificado no fue encontrado.")
    # except pd.errors.EmptyDataError:
        # logging.error("Error: El archivo CSV está vacío o tiene un formato incorrecto.")
    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}", exc_info=True)
