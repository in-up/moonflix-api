from distutils.command import config
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from recommender import item_based_recommendation, user_based_recommendation,_search_movies_r, _search_movies
from resolver import random_genres_items, random_items

origins = [
    # "http://localhost",
    # "http://localhost:3000",
    # "https://dq-hustlecoding.github.io/dqflex",
    # "https://dq-hustlecoding.github.io",
    # "http://api.dqflex.kro.kr:8080",
    # "http://api.dqflex.kro.kr",
    "*",
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(middleware=middleware)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/all/")
async def all_movies():
    result = await random_items()  # await 추가 
    return {"result": result}


@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result = await random_genres_items(genre)  # await 추가
    return {"result": result}



@app.get("/user-based/")
async def user_based(params: Optional[List[str]] = Query(None)):
    input_ratings_dict = dict(
        (int(x.split(":")[0]), float(x.split(":")[1])) for x in params
    )
    result = user_based_recommendation(input_ratings_dict)
    return {"result": result}


@app.get("/item-based/{item_id}")
async def item_based(item_id: str):
    result = item_based_recommendation(item_id)
    return {"result": result}

@app.get("/search/")
async def search_movies(query: str = Query(None)):
    result = await _search_movies(query)
    return {"result": result}

@app.get("/search-recommendation/") 
async def search_movies(query: str = Query(None)):
    result = await _search_movies_r (query)  # 변경된 함수명 사용 
    return {"result": result}