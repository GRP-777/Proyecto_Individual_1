import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


app = FastAPI()

# http://127.0.0.1:8000
 

# Leer el DataFrame
df_f1_2 = pd.read_csv('df_f1_2.csv')
df_f3 = pd.read_csv('df_f3.csv')
df_f3_4 = df_f3
df_f5 = pd.read_csv("df_f5.csv")
df_juegos_ml = pd.read_csv("df_juegos_ml.csv")


 

@app.get('/playtime_genre')
def PlayTimeGenre(genero):
    genero = genero.lower()
    genero = genero.capitalize()
    df_f1_2['release_date'] = pd.to_datetime(df_f1_2['release_date'], errors='coerce')
    max_horas_ano = None
    max_horas = 0
    horas_por_ano = {}
    
    for index, row in df_f1_2.iterrows():
        if genero in row['genres']:
            # Obtener el año de la fecha de lanzamiento
            year = row['release_date'].year
            
            # Sumar las horas jugadas
            horas_jugadas = row['playtime_forever']
            
            if year not in horas_por_ano:
                horas_por_ano[year] = 0
                
            horas_por_ano[year] += horas_jugadas
            
            if horas_por_ano[year] > max_horas:
                max_horas = horas_por_ano[year]
                max_horas_ano = year
    res = {
    "Año con más horas": max_horas_ano,"Total de horas sumadas": max_horas
    }
            
    return res



@app.get('/user_for_genre')
def UserForGenre(genero):
    genero = genero.lower()
    genero = genero.capitalize()
    df_f1_2['release_date'] = pd.to_datetime(df_f1_2['release_date'], errors='coerce')
    max_horas_ano = None
    max_horas = 0
    max_user = None  # Nuevo: mantener un registro del usuario con más horas jugadas
    horas_por_ano = {}
    
    for index, row in df_f1_2.iterrows():
        if genero in row['genres']:
            # Obtener el año de la fecha de lanzamiento
            year = row['release_date'].year
            
            # Sumar las horas jugadas
            horas_jugadas = row['playtime_forever']
            
            if year not in horas_por_ano:
                horas_por_ano[year] = 0
                
            horas_por_ano[year] += horas_jugadas
            
            if horas_por_ano[year] > max_horas:
                max_horas = horas_por_ano[year]
                max_horas_ano = year
                max_user = row['id']  # Nuevo: actualizar el usuario con más horas jugadas
    
    res = {
        "Año con más horas": max_horas_ano,
        "Total de horas sumadas": max_horas,
        "Usuario con más horas jugadas para Género X": max_user  # Nuevo: incluir el usuario
    }
            
    return res


@app.get('/users_recommend')
def UsersRecommend(año: int):
    # Verificar si el año es igual a -1 y mostrar un mensaje personalizado
    if año == -1:
        return "El año ingresado es -1, lo cual no es válido."

    # Verificar que el año sea un número entero
    if not isinstance(año, int):
        return "El año debe ser un número entero."

    # Verificar que el año ingresado esté en la columna 'year_integer'
    if año not in df_f3_4['year_integer'].unique():
        return "El año no se encuentra en la columna 'year_integer'."

    # Filtrar el dataset para obtener solo las filas correspondientes al año dado
    juegos_del_año = df_f3_4[df_f3_4['year_integer'] == año]

    # Calcular la cantidad de recomendaciones para cada juego
    recomendaciones_por_juego = juegos_del_año.groupby('item_name')['recommend'].sum().reset_index()

    # Ordenar los juegos por la cantidad de recomendaciones en orden descendente
    juegos_ordenados = recomendaciones_por_juego.sort_values(by='recommend', ascending=False)

    # Tomar los tres primeros lugares
    primer_puesto = juegos_ordenados.iloc[0]['item_name']
    segundo_puesto = juegos_ordenados.iloc[1]['item_name']
    tercer_puesto = juegos_ordenados.iloc[2]['item_name']

    # Crear el diccionario con los tres primeros lugares
    top_tres = {
        "Puesto 1": primer_puesto,
        "Puesto 2": segundo_puesto,
        "Puesto 3": tercer_puesto
    }

    return top_tres





@app.get('/users_not_recommend')
def UsersNotRecommend(año: int):
    # Verificar si el año es igual a -1 y mostrar un mensaje personalizado
    if año == -1:
        return "El año ingresado es -1, lo cual no es válido."

    # Verificar que el año sea un número entero
    if not isinstance(año, int):
        return "El año debe ser un número entero."

    # Verificar que el año ingresado esté en la columna 'year_integer'
    if año not in df_f3_4['year_integer'].unique():
        return "El año no se encuentra en la columna 'year_integer'."

    # Filtrar el dataset para obtener solo las filas correspondientes al año dado
    juegos_del_año = df_f3_4[df_f3_4['year_integer'] == año]

    # Calcular la cantidad de recomendaciones para cada juego
    recomendaciones_por_juego = juegos_del_año.groupby('item_name')['recommend'].sum().reset_index()

    # Ordenar los juegos por la cantidad de recomendaciones en orden descendente
    juegos_ordenados = recomendaciones_por_juego.sort_values(by='recommend', ascending=True)

    # Tomar los tres primeros lugares
    ultimo_puesto = juegos_ordenados.iloc[0]['item_name']
    penultimo_puesto = juegos_ordenados.iloc[1]['item_name']
    antepenultimo_puesto = juegos_ordenados.iloc[2]['item_name']

    # Crear el diccionario con los tres primeros lugares
    ultimos_tres = {
        "Puesto 1": ultimo_puesto,
        "Puesto 2": penultimo_puesto,
        "Puesto 3": antepenultimo_puesto
    }

    return ultimos_tres




@app.get('/sentiment_analysis')
def sentiment_analysis(año : int):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df_f5[df_f5['year'] == año]
    
    # Contar la cantidad de registros por cada análisis de sentimiento
    resultados = df_filtered['sentiment_analysis'].value_counts()
    
    # Si algún análisis de sentimiento está ausente, añádelo con 0
    for sentimiento in ['Negativo', 'Neutral', 'Positivo']:
        if sentimiento not in resultados:
            resultados[sentimiento] = 0
    
    # Almacenar los resultados en una lista de tuplas
    resultados_lista = [(sentimiento, cantidad) for sentimiento, cantidad in resultados.items()]
    
    return resultados_lista







# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Crear una instancia del codificador
label_encoder = LabelEncoder()

# Cargar los datos
df_ml = pd.read_csv("df_ml.csv")
df_ml["genres_encoded"] = label_encoder.fit_transform(df_ml["genres"])

# Crear un diccionario de los títulos asociados a cada item_id
titles_by_item_id = {}
for i in range(len(df_ml)):
    titles_by_item_id[df_ml.loc[i, "item_id"]] = df_ml.loc[i, "title"]

# Crear el modelo K-Nearest Neighbors
k = 5
model = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
model.fit(df_ml[['genres_encoded']], df_ml['title'])
 

@app.get('/obtener_recomendaciones')
# Crear una función para obtener las recomendaciones
def get_recommendations(item_id: int):
    # Buscar el género codificado del juego proporcionado por el usuario
    input_game = df_ml[df_ml["item_id"] == item_id]["genres_encoded"].values[0]

    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])

    # Obtener los títulos de los juegos similares
    similar_games = [titles_by_item_id[df_ml.loc[i, "item_id"]] for i in indices[0]]

    # Devolver un diccionario de los títulos
    return {"similar_games": similar_games}




'''
# Crear una instancia del codificador
label_encoder = LabelEncoder()
# Aplicar la codificación a la columna de géneros
df_juegos_ml['genres_encoded'] = label_encoder.fit_transform(df_juegos_ml['genres'])
# Supongamos que 'df' es tu DataFrame y 'genres_encoded' es la columna de géneros codificada
X = df_juegos_ml[['genres_encoded']]  # Características
y = df_juegos_ml['title']  # Objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Inicializa el modelo K-Nearest Neighbors
k = 5  # Número de vecinos
model = KNeighborsClassifier(n_neighbors=k)

# Entrena el modelo
model.fit(X_train, y_train)


@app.get('/obtener_recomendaciones')
def get_similar_games(titulo):
    titulo = titulo.lower()
    titulo = titulo.capitalize()
    # Buscar el género codificado del juego proporcionado por el usuario
    input_game = df_juegos_ml[df_juegos_ml['title'] == titulo]['genres_encoded'].values[0]
    
    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])
    
    # Obtener los títulos de los juegos similares en forma de diccionario invertido
    similar_games_dict = {df_juegos_ml.iloc[indices[0]]['title'].values[i]: i for i in range(len(indices[0]))}
    
    return similar_games_dict
    '''



'''

from sklearn.preprocessing import LabelEncoder

# Crear una instancia del codificador
label_encoder = LabelEncoder()

# Aplicar la codificación a la columna de géneros
df_juegos_ml['genres_encoded'] = label_encoder.fit_transform(df_juegos_ml['genres'])
# Supongamos que 'df' es tu DataFrame y 'genres_encoded' es la columna de géneros codificada
X = df_juegos_ml[['genres_encoded']]  # Características
y = df_juegos_ml['title']  # Objetivo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
# Inicializa el modelo K-Nearest Neighbors
k = 5  # Número de vecinos
model = KNeighborsClassifier(n_neighbors=k)
# Entrena el modelo
model.fit(X_train, y_train)

@app.get('/obtener_recomendaciones')
def get_similar_games(titulo):
    # Buscar el género codificado del juego proporcionado por el usuario
    titulo = titulo.lower()
    titulo = titulo.capitalize()
    input_game = df_juegos_ml[df_juegos_ml['title'] == titulo]['genres_encoded'].values[0]
    
    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])
    
    # Obtener los títulos de los juegos similares en forma de diccionario invertido
    similar_games_dict = {df_juegos_ml.iloc[indices[0]]['title'].values[i]: i for i in range(len(indices[0]))}
    
    return similar_games_dict
'''
