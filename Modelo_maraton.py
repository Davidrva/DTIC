
# importación de librerías necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Cargar los datos (supongo que el archivo se llama 'dataset.csv')
df = pd.read_csv('dataset.csv')

# Definir la función del modelo
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Ajustar el modelo a los datos
xdata = df['x'].values
ydata = df['y'].values

popt, pcov = curve_fit(model_func, xdata, ydata)

# Graficar los datos y la curva ajustada
plt.figure(figsize=(10, 6))
plt.scatter(xdata, ydata, label='Datos')
plt.plot(xdata, model_func(xdata, *popt), color='red', label='Ajuste: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

uploaded = files.upload()
import io
datos_maraton = pd.read_csv(io.BytesIO(uploaded['DatosMaratonenCSV.csv']))
datos_maraton
datos_maraton["Name"]
datos_maraton.info()
datos_maraton['Wall21'] = pd.to_numeric(datos_maraton['Wall21'],errors='coerce')
datos_maraton.describe()
datos_maraton.hist()
datos_maraton = datos_maraton.drop(columns=['Name'])
datos_maraton = datos_maraton.drop(columns=['id'])
datos_maraton = datos_maraton.drop(columns=['Marathon'])
datos_maraton = datos_maraton.drop(columns=['CATEGORY'])
datos_maraton
datos_maraton.isna().sum()
datos_maraton["CrossTraining"] = datos_maraton["CrossTraining"].fillna(0)
datos_maraton
datos_maraton = datos_maraton.dropna(how='any')
datos_maraton
datos_maraton['CrossTraining'].unique()
valores_cross = {"CrossTraining":  {'ciclista 1h':1, 'ciclista 3h':2, 'ciclista 4h':3, 'ciclista 5h':4, 'ciclista 13h':5}}
datos_maraton.replace(valores_cross, inplace=True)
datos_maraton
datos_maraton['CrossTraining'].unique()
datos_maraton['Category'].unique()
valores_categoria = {"Category":  {'MAM':1, 'M45':2, 'M40':3, 'M50':4, 'M55':5,'WAM':6}}
datos_maraton.replace(valores_categoria, inplace=True)
datos_maraton
import matplotlib.pyplot as plt
plt.scatter(x = datos_maraton['km4week'], y=datos_maraton['MarathonTime'])
plt.title('km4week Vs Marathon Time')
plt.xlabel('km4week')
plt.ylabel('Marathon Time')
plt.show()
plt.scatter(x = datos_maraton['sp4week'], y=datos_maraton['MarathonTime'])
plt.title('sp4week Vs Marathon Time')
plt.xlabel('sp4week')
plt.ylabel('Marathon Time')
plt.show()
datos_maraton = datos_maraton.query('sp4week<1000')
plt.scatter(x = datos_maraton['sp4week'], y=datos_maraton['MarathonTime'])
plt.title('sp4week Vs Marathon Time')
plt.xlabel('sp4week')
plt.ylabel('Marathon Time')
plt.show()
plt.scatter(x = datos_maraton['Wall21'], y=datos_maraton['MarathonTime'])
plt.title('Wall21 Vs Marathon Time')
plt.xlabel('Wall21')
plt.ylabel('Marathon Time')
plt.show()
datos_maraton
datos_entrenamiento = datos_maraton.sample(frac=0.8,random_state=0)
datos_test = datos_maraton.drop(datos_entrenamiento.index)
datos_entrenamiento
datos_test
etiquetas_entrenamiento = datos_entrenamiento.pop('MarathonTime')
etiquetas_test = datos_test.pop('MarathonTime')
etiquetas_entrenamiento
etiquetas_test
datos_entrenamiento
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(datos_entrenamiento,etiquetas_entrenamiento)
predicciones = modelo.predict(datos_test)
predicciones
import numpy as np
from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(etiquetas_test, predicciones))
print("Error porcentual : %f" % (error*100))
nuevo_corredor = pd.DataFrame(np.array([[1,400,20,0,1.4]]),columns=['Category', 'km4week','sp4week', 'CrossTraining','Wall21'])
nuevo_corredor
modelo.predict(nuevo_corredor)
