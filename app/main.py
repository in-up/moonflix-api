from distutils.command import config
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

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
    result = random_items()
    return {"result": result}


@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result = random_genres_items(genre)
    return {"result": result}
