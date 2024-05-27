import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import pickle

saved_model_fname = "model/finalized_model.sav"
data_fname = "data/ratings.csv"
item_fname = "data/movies_final.csv"
weight = 10


def model_train():
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")

    # create a sparse matrix of all the users/repos
    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),
            (
                ratings_df["movieId"].cat.codes.copy(),
                ratings_df["userId"].cat.codes.copy(),
            ),
        )
    )

    als_model = AlternatingLeastSquares(
        factors=50, regularization=0.01, dtype=np.float64, iterations=50
    )

    als_model.fit(weight * rating_matrix)

    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model


def calculate_item_based(item: str, movies_df, index):
    movies_df[item] = movies_df[item].fillna('')

    t = TfidfVectorizer(stop_words="english")
    tfidf_matrix = t.fit_transform(movies_df[item])
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similar_list = list(enumerate(cosine_matrix[int(index)]))

    return similar_list


def item_based_recommendation(title: str):
    movies_df = pd.read_csv(item_fname)

    title_to_index = movies_df[movies_df["title"]==title].index.values
    idx = title_to_index[0]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_overview = executor.submit(calculate_item_based, "overview", movies_df, idx)
        future_genres = executor.submit(calculate_item_based, "genres", movies_df, idx)
        future_title = executor.submit(calculate_item_based, "title", movies_df, idx)

        overview_similar_movies = future_overview.result()
        genres_similar_movies = future_genres.result()
        title_similar_movies = future_title.result()

    similar_movies = [(x1, y1 + y2 + (0.5 * y3)) for (x1, y1), (x2, y2), (x3, y3) in zip(overview_similar_movies, genres_similar_movies, title_similar_movies)]
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]
    
    movies_indexes = [index[0] for index in similar_movies]
    result_items = movies_df.iloc[movies_indexes].to_dict("records")

    return result_items


def calculate_user_based(user_items, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.recommend(
        userid=0, user_items=user_items, recalculate_user=True, N=10
    )
    return [str(items[r]) for r in recs[0]]


def build_matrix_input(input_rating_dict, items):
    model = pickle.load(open(saved_model_fname, "rb"))
    # input rating list : {1: 4.0, 2: 3.5, 3: 5.0}

    item_ids = {r: i for i, r in items.items()}
    mapped_idx = [item_ids[s] for s in input_rating_dict.keys() if s in item_ids]
    data = [weight * float(x) for x in input_rating_dict.values()]
    # print('mapped index', mapped_idx)
    # print('weight data', data)
    rows = [0 for _ in mapped_idx]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()


def user_based_recommendation(input_ratings):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    input_matrix = build_matrix_input(input_ratings, items)
    result = calculate_user_based(input_matrix, items)
    result = [int(x) for x in result]
    result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
    return result_items

# 검색 영화와 비슷한 장르의 영화 추천
async def _search_movies_r(query):
    if not query:
        return []

    movies_df = pd.read_csv(item_fname)
    movies_df['title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    search_terms = query.lower().split()

    # 검색어 전체 포함 영화 찾기
    full_match_mask = movies_df['title'].str.lower() == query.lower()
    full_match_df = movies_df[full_match_mask]

    # 전체 일치 결과가 없으면 단어 단위 검색
    if full_match_df.empty:
        # 검색어 포함 영화 찾기 (단어 단위)
        mask = movies_df['title'].str.lower().str.contains('|'.join(search_terms), regex=False)
        filtered_df = movies_df[mask]
    else:
        filtered_df = full_match_df
    # 결과 합치기 (중복 제거 포함)
    result_df = pd.concat([filtered_df, full_match_df])
    result_df = result_df.drop_duplicates(subset=['movieId'])

   # 검색 결과가 있는 경우에만 장르 마스크 생성
    if not result_df.empty:
        search_genres = result_df['genres'].tolist()[0].split('|') 

    # 시리즈 영화 찾기
        series_titles = []
        for title in result_df['title']: 
            base_title = title.split(':')[0].strip()
            series_mask = movies_df['title'].str.startswith(base_title)
            series_titles.extend(movies_df[series_mask]['title'].tolist())
            
        # 유사 장르 영화 추천 (Top 5)
        genre_similarity = []
        for idx, row in movies_df.iterrows():
            if row['title'] not in series_titles and row['title'] not in result_df['title'].tolist():
                movie_genres = row['genres'].split('|')
                common_genres = len(set(search_genres) & set(movie_genres))  
                genre_similarity.append((idx, common_genres))

        genre_similarity.sort(key=lambda x: x[1], reverse=True)  
        top_similar_indices = [idx for idx, count in genre_similarity[:5]] 

        recommended_df = movies_df.iloc[top_similar_indices].sort_values(by='rating_count', ascending=False)  # rating_count 기준 내림차순 

        # 결과 합치기 (중복 제거 포함)
        result_df = pd.concat([result_df, 
                               movies_df[movies_df['title'].isin(series_titles)], 
                               recommended_df])
        result_df = result_df.drop_duplicates(subset=['movieId'])
        result_items = result_df.to_dict('records')

    return result_items

# 검색한 영화의 시리즈까지만 출력
async def _search_movies(query):
    if not query:
        return []

    movies_df = pd.read_csv(item_fname)
    movies_df['title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    search_terms = query.lower().split()

    full_match_mask = movies_df['title'].str.lower() == query.lower()
    full_match_df = movies_df[full_match_mask]

    if full_match_df.empty:
        mask = movies_df['title'].str.lower().str.contains('|'.join(search_terms), regex=False)
        filtered_df = movies_df[mask]
    else:
        filtered_df = full_match_df
    result_df = pd.concat([filtered_df, full_match_df])
    result_df = result_df.drop_duplicates(subset=['movieId'])
    
    if not result_df.empty: 
        genre_mask = movies_df['genres'].str.contains('|'.join(result_df['genres'].tolist()[0].split('|')))
        
        series_titles = []
        for title in result_df['title']: 
            base_title = title.split(':')[0].strip()
            series_mask = movies_df['title'].str.startswith(base_title)
            series_titles.extend(movies_df[series_mask]['title'].tolist())
            
            # 결과 합치기 (중복 제거 포함)
        result_df = pd.concat([result_df, 
                               movies_df[movies_df['title'].isin(series_titles)], 
                               ])
        result_df = result_df.drop_duplicates(subset=['movieId'])
            
    result_items = result_df.to_dict('records')
    
    return result_items
    
if __name__ == "__main__":
    model = model_train()