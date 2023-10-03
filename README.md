# Henry Individual Project 1
# Data Science & Machine Learning Operations (MLOps)

![MLOps](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/c5259852-e96b-439c-a1af-f89124128043)

# Introduction (context and tasks)

The current project it's based in fiction data and it's not a project seeking for results of an analysis, but to show some abilities I'm able to perform.

I was entrusted with the assignment of developing an API using the **FastAPI** framework to show a gaming database analysis and recommendation system. The asked result was a _**Minimum Viable Product (MVP)**_ containing 5 function endpoints and s last one for a machine learning recommendation system.

![PI1_MLOps_Mapa1](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/f36720bf-8322-48a0-a002-95dd2acc1944)


# Dataset Description and Dictionary
To download the original datasets, due to their weight, they can be found at the following link. [Original Datasets](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)


| **Column**         | **Description**                                                   | **Example**                                                                                                                                                                           |
|------------------- |------------------------------------------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| publisher          | Content publisher                                                  | [Ubisoft, Dovetail Games - Trains, Degica]                                                                                                                                           |
| genres             | Content genre                                                      | [Action, Adventure, Racing, Simulation, Strategy]                                                                                                                                    |
| app_name           | Content name                                                       | [Warzone, Soundtrack, Puzzle Blocks]                                                                                                                                                 |
| title              | Content title                                                      | [The Dream Machine: Chapter 4 , Fate/EXTELLA - Sweet Room Dream, Fate/EXTELLA - Charming Bunny]                                                                                       |
| url                | Content publication URL                                            | [http://store.steampowered.com/app/761140/Lost_Summoner_Kitty/]                                                                                                                      |
| release_date       | Release date                                                       | [2018-01-04]                                                                                                                                                                         |
| tags               | Content tags                                                       | [Simulation, Indie, Action, Adventure, Funny, Open World, First-Person, Sandbox, Free to Play]                                                                                       |
| discount_price     | Discount price                                                     | [22.66, 0.49, 0.69]                                                                                                                                                                  |
| reviews_url        | Content reviews URL                                                | [http://steamcommunity.com/app/681550/reviews/?browsefilter=mostrecent&p=1]                                                                                                           |
| specs              | Specifications                                                     | [Multi-player, Co-op, Cross-Platform Multiplayer, Downloadable Content]                                                                                                              |
| price              | Content price                                                      | [4.99, 9.99, Free to Use, Free to Play]                                                                                                                                              |
| early_access       | Early access                                                       | [False, True]                                                                                                                                                                        |
| id                 | Unique content identifier                                          | [761140, 643980, 670290]                                                                                                                                                            |
| developer          | Developer                                                          | [Kotoshiro, Secret Level SRL, Poolians.com]                                                                                                                                         |
| metascore          | Metacritic score                                                   | [80, 74, 77, 75]                                                                                                                                                                    |
| user_id            | Unique user identifier                                             | [76561197970982479, evcentric, maplemage]                                                                                                                                            |
| user_url           | User profile URL                                                   | [http://steamcommunity.com/id/evcentric]                                                                                                                                             |
| reviews            | User review in Json format                                         | {'funny': '', 'posted': 'Posted September 8, 2013.','last_edited': '','item_id': '227300','helpful': '0 of 1 people (0%) found this review helpful','recommend': True,'review': "For a simple..."}                                       |                                                                                                                                                                                   |
| user_id            | Unique user identifier                                             | [76561197970982479, evcentric, maplemage]                                                                                                                                            |
| user_url           | User profile URL                                                   | [http://steamcommunity.com/id/evcentric]                                                                                                                                             |
| items              | User items in Json format                                          | {'item_id': '273350', 'item_name': 'Evolve Stage 2', 'playtime_forever': 58, 'playtime_2weeks': 0}                                                                                |



Processes
ETL:
To find out more about the development of the ETL process, there is the following link
[ETL Documentation](https://github.com/GRP-777/Proyecto_Individual_1/blob/master/PI_ML_Ops_ETL.ipynb)

_**Datasets names**_:
- australian_user_reviews
- australian_users_items
- output_steam_games

_**Unnest**_:
1. Some columns are nested, that is they either have a dictionary or a list as values ​​in each row, we unnest them to be able to do some of the API queries.

_**Drop unused columns**_:
2. We remove the columns that will not be used:
   - From the output_steam_games: publisher, app_name, discount_price, tags, early_access, specs, price, metascore, developer, items_count, reviews_url, steam_id, playtime_2weeks, url and Unnamed: 0.
   - From the australian_user_reviews: last_edited, user_url, Unnamed:0, helpful and funny.
   - From the australian_users_items: items_count, review, steam_id, playtime_2weeks, user_id and Unnamed: 0.

_**Control of null values**_:
3. There are null values in:
   - From the output_steam_games: genres, release_date, and id.
   - From the australian_user_reviews: posted, item_id, review, and recommend.

_**Daytime datatype arrangement**_:
4. The dates are changed to year integers:
   - From the australian_user_reviews: posted column.
   - From the output_steam_games: release_date column.

_**Duplicates dropping**_:
5. Duplicated ids usualy affect the results:
   - From the australian_users_items: item_ids.

_**Datasets merging**_:
6. Combine the two cleaned datasets: australian_users_items and output_steam_games.

_**Sentiment analysis**_:
7. In the australian_user_reviews dataset, there are reviews of games made by different users. Creation of the column 'sentiment_analysis' by applying NLP sentiment analysis with the following scale: it takes the value '0' if it's negative, '1' if it's neutral, and '2' if it's positive. This new column replaces the australian_user_reviews.review column to facilitate the work of the machine learning models and data analysis. If this analysis is not possible due to the absence of a written review, it takes the value of 1.


# _Functions_
- _**For more information about the development of the different functions and a more detailed explanation of each one, please click the following link.**_
[Functions Notebook](https://github.com/GRP-777/Proyecto_Individual_1/blob/master/main.py)

API Development: The proposal is to make a company's data available using the FastAPI framework. This framework is built on modern Python 3.7+ features such as function data types and type annotations, which allow for high productivity and cleaner, more readable code.

Each app one has a decorator (@app.get('/')) with the name of the app in it to be able to recognize it's function.

The queries are as follows:

1. PlayTimeGenre(genre):
Should return the year with the most hours played for the given genre.
Example return: {"Year with the most hours played for Genre X": 2013}

2. UserForGenre(genre):
Should return the user with the highest accumulated playtime for the given genre and a list of playtime accumulation by year.
Example return: {"User with the most playtime for Genre X": "us213ndjss09sdf", "Playtime": [{"Year": 2013, "Hours": 203}, {"Year": 2012, "Hours": 100}, {"Year": 2011, "Hours": 23}]}

3. UsersRecommend(year: int):
Returns the top 3 games MOST recommended by users for the given year. (reviews.recommend = True and positive/neutral comments)
Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

4. UsersNotRecommend(year: int):
Returns the top 3 games LEAST recommended by users for the given year. (reviews.recommend = False and negative comments)
Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

5. sentiment_analysis(year: int):
Based on the release year, returns a list with the count of user review records categorized with sentiment analysis.
Example return: {"Negative": 182, "Neutral": 120, "Positive": 278}
Exploratory Data Analysis & Machine Learning

# _**Exploratory Data Analysis & Machine Learning**_

The model establishes an item-item relationship. This means that given an item_id, based on how similar it is to the rest, similar items will be recommended. Here, the input is a game and the output is a list of recommended games. The machine learning method used is K-Neighbours. It's not the best method to approach to the datasets, and part of this project it's focused on that. Because the project needs to be deployed on Render, the RAM memory available is limited and the importance here was to understand the difference between the different Machine Learning models. Previously, I tried decision trees and natural languaje proccesing using cosine similarity.

It's an item-item recommendation system:
6. recommendation_game(product_id): 
By inputting the product ID, we should receive a list of 5 recommended games similar to the one provided.



API Deployment
The deployment of our FastAPI is done using Render a virtual environment.

To consume the API, use the 6 different endpoints to get information and make queries about gaming stadistics.
![endpoints](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/90fee9b4-101b-458e-8521-daa18720edfb)



# Requirements
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- FastAPI
- [Render](https://render.com/)
# _Author_
Germán Robles Pérez
Mail: groblesperez0@gmail.com
Linkedin: [https://www.linkedin.com/in/fgc97/](https://www.linkedin.com/in/germ%C3%A1n-robles-p%C3%A9rez-4298b71b3/)
