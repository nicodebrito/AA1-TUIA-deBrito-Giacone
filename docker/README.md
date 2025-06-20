# Modelo de prediccion de lluvias en Australia con Docker

Este proyecto tiene como objetivo, a partir de observaciones que sirven como datos de entrada, realizar las transformaciones e imputaciones necesarias para predecir si lloverá al día siguiente. 

Para ello, se utiliza una tubería de procesamiento (`pipeline.pkl`) que incluye tanto los pasos de preprocesamiento como un modelo de red neuronal entrenado. 

Todo el proceso de inferencia se ejecuta dentro de un contenedor Docker.


Autores:
* de Brito, Nicolas
* Giacone, Agustin

---

## 🚀 Instrucciones

### 1. Abrir una terminal en la carpeta raíz del proyecto

### 2. Construir la imagen de Docker

```bash
docker build -t inference-python-test ./docker
```

> Esta instrucción crea una imagen Docker llamada `inference-python-test` a partir del `Dockerfile` ubicado en la carpeta `./docker`.

### 3. Ejecutar el contenedor

```bash
docker run -it --rm --name inference-python-test -v ./files:/files inference-python-test
```

> Este comando monta la carpeta local `./files` dentro del contenedor en la ruta `/files`, donde se espera el archivo de entrada y se generará la salida.

---

## ⚠️ Nota para usuarios de Windows

En sistemas Windows puede ser necesario usar la ruta **absoluta** para montar el volumen. Si la consola arroja un error con el comando anterior, pruebe con el siguiente formato:

```bash
docker run -it --rm --name inference-python-test -v "UnidadAlmacenamiento:\rutaCarpetaDelProyecto\docker\files:/files" inference-python-test
```

Reemplace:

- `UnidadAlmacenamiento` → la letra de unidad correspondiente (por ejemplo, `C`)
- `rutaCarpetaDelProyecto` → la ruta completa desde la raíz de la unidad hasta la carpeta `docker\files`

🔎 **Ejemplo:**

Si su carpeta se encuentra en:

```
C:\Users\JuanPerez\Desktop\carpetaProyecto
```

El comando sería:

```bash
docker run -it --rm --name inference-python-test -v "C:\Users\JuanPerez\Desktop\carpetaProyecto\docker\files:/files" inference-python-test
```

---

## 📁 Requisitos de Archivos

- El script espera encontrar un archivo en la carpeta `files` llamado:  
  `input.csv`
- El script genera y guarda el archivo:  
  `output.csv` en la misma carpeta `files`

---

## 📝 Formato del archivo `input.csv`

El archivo de entrada (`input.csv`) debe contener una o más filas con los siguientes campos, separados por comas (formato CSV con encabezado en la primera fila):

```csv
Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday
```

🔹 Todas las columnas son necesarias para que el pipeline funcione correctamente.

🔹 El campo `Date` debe estar en formato `YYYY-MM-DD` (por ejemplo: `2021-06-15`).

🔹 El campo `RainToday` debe tener los valores `'Yes'` o `'No'`.

🔹 El campo Location hace referencia a una ciudad australiada. Debe estar incluida en el siguiente listado:

Adelaide, Albany, Albury, BadgerysCreek, Ballarat, Bendigo, Brisbane, Cairns, Canberra, Cobar, CoffsHarbour, Dartmoor, Darwin, GoldCoast, Hobart, Katherine, Launceston, Melbourne, MelbourneAirport, Mildura, Moree, MountGambier, MountGinini, Newcastle, Nhil, NorahHead, NorfolkIsland, Nuriootpa, PearceRAAF, Penrith, Perth, PerthAirport, Portland, Richmond, Sale, Sydney, SydneyAirport, Townsville, Tuggeranong, Uluru, WaggaWagga, Walpole, Watsonia, Williamtown, Witchcliffe, Wollongong, Woomera

---

### 📌 Ejemplo de una fila válida

```csv
2021-06-15,Sydney,7.4,19.5,0.0,5.6,9.8,W,41,W,SW,20,24,68,55,1015.1,1012.3,4,4,12.3,18.5,No
```

> Si hay valores faltantes, dejarlos en blanco (es decir, dos comas seguidas), el pipeline se encargará de imputarlos.

---

## Resultados

Los resultados se generan automáticamente en `files/output.csv` con el formato:

```csv
RainTomorrow_predicted
1
1
0
```

Donde:
- `0` = No lloverá mañana
- `1` = Lloverá mañana

## Estructura de Archivos

```
docker/
├── files/
│   ├── input.csv          # Datos de entrada
│   └── output.csv         # Predicciones
├── Dockerfile
├── inference.py
├── pipeline.pkl
├── requirements.txt
└── clases_pipeline.py
```

## ✅ Requisitos previos

- Tener [Docker](https://www.docker.com/products/docker-desktop/) instalado y funcionando
- Tener el archivo `pipeline.pkl` incluido dentro de la imagen
- Tener un archivo `input.csv` preparado y ubicado en la carpeta `files`

---


