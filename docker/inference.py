import joblib
import pandas as pd
import warnings
import sklearn
print(sklearn.__version__)
warnings.simplefilter('ignore')

import logging
from sys import stdout

from clases_pipeline import (
    TransformRainToday,
    ImputeRainfall,
    MergeAndImputeCuantitativas,
    Transformaciones,
    Escalado,
    MergeAndImputeCualitativas,
    MergeAndTransformLocation,
    NeuralNetworkTensorFlow
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

#pipeline = joblib.load('./docker/pipeline.pkl')
pipeline = joblib.load('pipeline.pkl')

logger.info('Pipeline cargado')

#df_input = pd.read_csv('./docker/files/input.csv') #C:/Users/nicod/OneDrive/Desktop/test
df_input = pd.read_csv('./files/input.csv')

logger.info('Informacion input cargada')
logger.info('Visualizacion de primeros 5 registros input')
print(df_input.head(5))


clean_input = pipeline[:-1].transform(df_input)
logger.info('Transformacion e imputacion de datos')
y_prob = pipeline[-1].predict(clean_input)
logger.info('Probabilidades generadas')
output = (y_prob >= 0.3).astype(int)
logger.info('Predicciones generadas')

#print(output)
#pd.DataFrame(output, columns=['RainTomorrow_predicted']).to_csv('./docker/files/output.csv', index=False) #C:/Users/nicod/OneDrive/Desktop/test
#pd.DataFrame(output, columns=['RainTomorrow_predicted']).to_csv('./files/output.csv', index=False)

with open('/files/output.csv', 'w') as f:
    pd.DataFrame(output, columns=['RainTomorrow_predicted']).to_csv(f, index=False)

logger.info('Archivo con predicciones generado y guardado en ./docker/files/output.csv')