import pandas
import requests

if __name__ == "__main__":
    movies_df = pandas.read_csv('data/movies.csv')
    print(movies_df)
