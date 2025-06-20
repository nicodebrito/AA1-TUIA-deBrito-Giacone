from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TransformRainToday(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        X = X.copy()
        X['RainToday'] = X['RainToday'].map({'Yes': 1, 'No': 0})
        return X

class ImputeRainfall(BaseEstimator, TransformerMixin):
    def __init__(self, mediana_Rainfall_train):
        self.mediana_Rainfall_train = mediana_Rainfall_train

    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        X = X.copy()
        X['Rainfall'] = X['Rainfall'].fillna(self.mediana_Rainfall_train)
        return X

class MergeAndImputeCuantitativas(BaseEstimator, TransformerMixin):
    def __init__(self, imputadores_dict):
        self.imputadores_dict = imputadores_dict

    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        #print('Copiando df\n')
        X_test = X.copy()
        #print('Cambiando a tipo de dato date \n')
        X_test['Date'] = pd.to_datetime(X_test['Date'], errors='coerce').dt.date
        #print(X_test['Date'].dtype)
        #print('Creando year \n')
        X_test['Year'] = pd.to_datetime(X_test['Date'], errors='coerce').dt.year
        X_test['Month'] = pd.to_datetime(X_test['Date'], errors='coerce').dt.month
        for col , imputador in self.imputadores_dict.items():
          #print('Se esta usando el dataframe', imputador[0] ,'para crear la columna',col+'_mediana...')
          X_test = X_test.merge(imputador[1], on=['Location', 'Year', 'Month'], how='left', suffixes=('', '_mediana'))
          #print('Se van a imputar',X_test[X_test[col].isna()].shape[0],'valores faltantes')
          #print('Se esta imputando los faltantes de la columna ',col, 'con la columna ',col+'_mediana','\n\n')
          X_test[col] = X_test[col].fillna(X_test[col+'_mediana'])
          #print('Cantidad de valores NaN en ',col,':',X_test[X_test[col].isna()].shape[0])

        return X_test

class Transformaciones(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        #print('Copiando df\n')
        X_test = X.copy()
        #Seteo grados a cada direccion
        puntos_cardinales = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        X_test['WindGustDir_grad'] = X_test['WindGustDir'].map(puntos_cardinales)
        X_test['WindDir9am_grad'] = X_test['WindDir9am'].map(puntos_cardinales)
        X_test['WindDir3pm_grad'] = X_test['WindDir3pm'].map(puntos_cardinales)

        # Convertimos los grados a radianes
        X_test['WindGustDir_rad'] = np.deg2rad(X_test['WindGustDir_grad'])
        X_test['WindDir9am_rad'] = np.deg2rad(X_test['WindDir9am_grad'])
        X_test['WindDir3pm_rad'] = np.deg2rad(X_test['WindDir3pm_grad'])

        # Creamos las variables cíclicas
        X_test['WindGustDir_sin'] = np.sin(X_test['WindGustDir_rad']).round(5)
        X_test['WindGustDir_cos'] = np.cos(X_test['WindGustDir_rad']).round(5)

        X_test['WindDir9am_sin'] = np.sin(X_test['WindDir9am_rad']).round(5)
        X_test['WindDir9am_cos'] = np.cos(X_test['WindDir9am_rad']).round(5)

        X_test['WindDir3pm_sin'] = np.sin(X_test['WindDir3pm_rad']).round(5)
        X_test['WindDir3pm_cos'] = np.cos(X_test['WindDir3pm_rad']).round(5)

        X_test['month_sin'] = np.sin(2 * np.pi * X_test['Month'] / 12).round(5)
        X_test['month_cos'] = np.cos(2 * np.pi * X_test['Month'] / 12).round(5)

        columnas_a_escalar = [
          'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',  'WindSpeed9am',
          'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
          'Pressure3pm', 'Temp9am', 'Temp3pm', 'month_sin', 'month_cos',
          'WindGustDir_sin', 'WindGustDir_cos', 'WindDir9am_sin', 'WindDir9am_cos',
          'WindDir3pm_sin', 'WindDir3pm_cos']

        columnas_sin_escalar = [
          'zona_2', 'zona_3', 'zona_4','zona_5', 'zona_6', 'zona_7', 'zona_8','RainToday']

        #X_test = X_test[columnas_a_escalar + columnas_sin_escalar]
        return X_test


class Escalado(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, columnas_a_escalar):
        self.scaler = scaler 
        self.columnas_a_escalar = columnas_a_escalar

    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        
        columnas_escaladas = [col + '_escalada' for col in self.columnas_a_escalar]
        columnas_sin_escalar = [
            'zona_2', 'zona_3', 'zona_4','zona_5', 'zona_6', 'zona_7', 'zona_8','RainToday']
        
        X_test = X.copy()
        X_test_scaled = self.scaler.transform(X_test[self.columnas_a_escalar])
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=columnas_escaladas, index=X_test.index)
        X_test = pd.concat([X_test, X_test_scaled_df], axis=1)
        X_test = X_test[columnas_escaladas + columnas_sin_escalar]
        return X_test


class MergeAndImputeCualitativas(BaseEstimator, TransformerMixin):
    def __init__(self, imputadores_dict_cualitativas):
        self.imputadores_dict_cualitativas = imputadores_dict_cualitativas

    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        X_test = X.copy()
        for col , imputador in self.imputadores_dict_cualitativas.items():
          #print('Se esta usando el dataframe', imputador ,'para crear la columna',col+'_mediana...')
          X_test = X_test.merge(imputador, on=['Location', 'Year', 'Month'], how='left', suffixes=('', '_moda'))
          #print('Se van a imputar',X_test[X_test[col].isna()].shape[0],'valores faltantes')
          #print('Se esta imputando los faltantes de la columna ',col, 'con la columna ',col+'_mediana','\n\n')
          X_test[col] = X_test[col].fillna(X_test[col+'_moda'])
          #print('Cantidad de valores NaN en ',col,':',X_test[X_test[col].isna()].shape[0])

        return X_test


class MergeAndTransformLocation(BaseEstimator, TransformerMixin):
    def __init__(self, imputador_location):
        self.imputador_location = imputador_location

    def fit(self, X, y=None):
        return self  # no hay nada que entrenar

    def transform(self, X):
        X_test = X.copy()
        X_test = X_test.merge(self.imputador_location, on=['Location'], how='left')
        #print(X_test['clim_zone'])
        X_test['clim_zone'] = X_test['clim_zone'].astype(int)
        
        zonas = ['zona_2', 'zona_3', 'zona_4','zona_5', 'zona_6', 'zona_7', 'zona_8']
        X_test[zonas] = 0

        for i in range(2, 9):
          X_test.loc[X_test['clim_zone'] == i, f'zona_{i}'] = 1

        return X_test


class NeuralNetworkTensorFlow:
    """
        (1) se construye el modelo.
        (2) Se define como se fitea el modelo
        (3) Y como se hacen las predicciones.
    """
    def __init__(self, learning_rate=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        """
            Construye el modelo
            Para construir el modelo es necesario una arquitectura, un optimizador y una función de pérdida.
            La arquitectura se construye con el método Sequential, que basicamente lo que hace es colocar
            secuencialmente las capas que uno desea.
            Las capas "Dense" son las fully connected dadas en clase.
            Se agrega una capa oculta que recibe un input de tamaño 2,
            y una capa de salida de regresión (una única neurona)
            En todos los casos se define una sigmoidea como función de activación (prueben otras!)

            El optimizador y la función de pérdida se especifican dentro de un compilador.

            Con este método, lo que se devuelve es el modelo sin entrenar, sería equivalente a escribir LinearRegression()
            en el caso de la regresión lineal.
        """
        model = Sequential([
            Dense(16, activation='relu', input_shape=(28,)),
            Dropout(0.5),
            Dense(12, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='tanh'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=SGD(learning_rate=self.learning_rate), loss='binary_crossentropy') #mse
        ### imprimimos la cantidad de parámetros a modo de ejemplo
        print("n° de parámetros:", model.count_params())
        return model

    def fit(self, X, y):
        ### esta es la función donde se entrena el modelo, fijarse que hay un learning rate e iteraciones.
        ### la función que fitea devuelve una historia de pérdida, que vamos a guardar para graficar la evolución.
        X = np.array(X)
        y = np.array(y)
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        return history.history['loss']

    def predict(self, X):
        X = np.array(X)
        predictions = self.model.predict(X)
        return predictions
