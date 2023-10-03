import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# API Development: You propose making the company's data available using the FastAPI framework. The queries you suggest are as follows:

app = FastAPI()

# You must create the following functions for the endpoints that will be consumed in the API. Remember that each one should have a decorator (@app.get('/')).


 
# DataFrames
df_f1_2 = pd.read_csv('df_f1_2.csv')
df_f3 = pd.read_csv('df_f3.csv')
df_f3_4 = df_f3
df_f5 = pd.read_csv("df_f5.csv")
df_juegos_ml = pd.read_csv("df_juegos_ml.csv")


@app.get("/")
def Presentation():
    """Welcome to my first Henry project"""
    return {"Germán Robles Pérez": "Welcome to my first Henry project. To explore the API documentation, please visit https://project1-henry.onrender.com/docs."}


#Endpoint 1

#PlayTimeGenre(genre):
#Should return the year with the most hours played for the given genre.
#Example return: {"Year with the most hours played for Genre X": 2013}



@app.get('/playtime_genre')
def PlayTimeGenre(genero):
 
    """The playtime_genre function takes a language (string) as input, validates its format, and returns a dictionary. The dictionary contains the year with the most hours played for the given genre. If the year it's not valid, it will throw an error."""
   
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



#UserForGenre(genre):
#Should return the user with the highest accumulated playtime for the given genre and a list of playtime accumulation by year.
#Example return: {"User with the most playtime for Genre X": "us213ndjss09sdf", "Playtime": [{"Year": 2013, "Hours": 203}, {"Year": 2012, "Hours": 100}, {"Year": 2011, "Hours": 23}]}

@app.get('/user_for_genre')
def UserForGenre(genero):

    """The user_for_genre function takes a language (string) as input, validates its format, and returns a dictionary. The dictionary contains the highest accumulated playtime for the given genre. If the year it's not valid, it will throw an error."""
   
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



#UsersRecommend(year):
#Returns the top 3 games MOST recommended by users for the given year. (reviews.recommend = True and positive/neutral comments)
#Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

@app.get('/users_recommend')
def UsersRecommend(year: int):
    
    """The users_recommend function takes a year (integer) as input, validates its format, and returns a dictionary. The dictionary contains the top 3 games MOST recommended by users for the given year. If the year it's not valid, it will throw an error."""
   
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



#UsersNotRecommend(year):
#Returns the top 3 games LEAST recommended by users for the given year. (reviews.recommend = False and negative comments)
#Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

@app.get('/users_not_recommend')
def UsersNotRecommend(año: int):
     
    """The users_not_recommend function takes a year (integer) as input, validates its format, and returns a dictionary. The dictionary contains the top 3 games LEAST recommended by users for the given year. If the year it's not valid, it will throw an error."""
  
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




#sentiment_analysis(year):
#Based on the release year, returns a list with the count of user review records categorized with sentiment analysis.
#Example return: {"Negative": 182, "Neutral": 120, "Positive": 278}

@app.get('/sentiment_analysis')
def sentiment_analysis(año : int):
      
    """The sentiment_analysis function takes a year (integer) as input, validates its format, and returns a list. Based on the release year, returns a list with the count of user review records categorized with sentiment analysis. If the year it's not valid, it will throw an error."""
  
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




#Function recommendation_game(product_id): By inputting the product ID, we receive a list of 5 recommended games similar to the one provided.

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
 

@app.get('/get_recommendations')
# Crear una función para obtener las recomendaciones
def get_recommendations(item_id: int):
      
    """The get_recommendations function takes a item_id (integer) as input, validates its format, and returns a list. We receive a list of 5 recommended games similar to the one provided. If the year it's not valid, it will throw an error."""
  
    # Buscar el género codificado del juego proporcionado por el usuario
    input_game = df_ml[df_ml["item_id"] == item_id]["genres_encoded"].values[0]

    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])

    # Obtener los títulos de los juegos similares
    similar_games = [titles_by_item_id[df_ml.loc[i, "item_id"]] for i in indices[0]]

    # Devolver un diccionario de los títulos
    return {"similar_games": similar_games}


