import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from collections import defaultdict
import random
import warnings
import logging

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")
random.seed(5153)
logging.basicConfig(level=logging.DEBUG)


class Model:
    def __init__(self):
        self.cache_path = "Data/cache.pkl"
        self.is_loaded = False
        self.dataset = None
        self.predictions = None
        self.user_details = None
        self.temp_store = None
        self.pipeline = None
        self.chosen_books_per_user = None
        self.all_books = pd.read_csv("Data/books.csv")
        logging.info("Initialized model")

    def run_predictions_on_full_test(self):
        if self.is_loaded:
            logging.info("Model is already loaded")
            return
        if self.does_cache_exist():
            logging.info("Retrieving cached full-test predictions")
            self.retrieve_cache()
            logging.info("Completed full-test")
            return
        logging.info("Generating full-test predictions")
        reviews_df = pd.read_csv("Data/final_dataset/reviews_test.csv")
        good_reviews = reviews_df[reviews_df['rating'] > 3]
        good_user_books_dict = good_reviews.groupby('user_id')['book_id'].unique().apply(list).to_dict()

        # to further minimize compute time, we only use 20 (randomly sampled) users
        num_random_users = 20
        randomly_sampled_users = random.sample(list(good_user_books_dict.keys()), num_random_users)
        sampled_good_user_books_dict = {user_id: good_user_books_dict[user_id] for user_id in randomly_sampled_users}

        # to minimize compute time, we take only 150 random (good) books per user
        # prepare it in the form of user_id -> list[book_id]
        num_rand_books_per_user = 150
        chosen_books_per_user = {
            user_id: random.sample(books, min(len(books), num_rand_books_per_user))
            for user_id, books in sampled_good_user_books_dict.items()
        }

        # save this for reference
        self.chosen_books_per_user = chosen_books_per_user

        # run predictions on all of the above users
        self.prepare_predictions(chosen_books_per_user)
        logging.info("Caching full-test predictions")
        self.cache_results()
        logging.info("Completed full-test")

    def run_prediction_on_adhoc_user(self, chosen_book_ids):
        self.prepare_predictions(
            {'current_user': chosen_book_ids}
        )

    def prepare_predictions(self, target_users_and_books):
        """
        Given a dictionary of user_id to list[book_id], where the list of book IDs are the books favored by
        the associated user, this function returns the recommended books for each user provided in the dictionary

        :param target_users_and_books: Dictionary of user ID to favored books (as book IDs)
        :return: Dataframe of user IDs and associated recommended books, plus individual model scores
        """
        target_user_list = list(target_users_and_books.keys())

        file_dict = {}
        for filename in ['reviews_test', 'users_test', 'reviews_sub']:
            file_dict[filename] = pd.read_csv(f'Data/final_dataset/{filename}.csv')

        file_dict['users'] = file_dict['users_test']
        file_dict['reviews'] = file_dict['reviews_test']

        file_dict['good_reviews'] = file_dict['reviews'][file_dict['reviews']['rating'] > 3]
        file_dict['books'] = pd.read_csv('Data/books.csv')

        #################################################################################
        # GENRE MODEL; DESCRIPTION MODEL; TITLE MODEL; BOOK STATS CLUSTER MODEL
        #################################################################################

        clusterbooks = pd.DataFrame(
            np.load('Data/Recommended Storage/cluster_books.npy', allow_pickle=True),
            columns=['target_book', 'recco_book_id', 'similarity_score']).astype(float)  # wasn't saved as float
        genrebooks = pd.DataFrame(
            np.load('Data/Recommended Storage/genres_books.npy', allow_pickle=True),
            columns=['target_book', 'recco_book_id', 'similarity_score'])
        descbooks = pd.DataFrame(
            np.load('Data/Recommended Storage/description_books.npy', allow_pickle=True),
            columns=['target_book', 'recco_book_id', 'similarity_score'])
        revbooks = pd.DataFrame(
            np.load('Data/Recommended Storage/reviews_books_new.npy', allow_pickle=True),
            columns=['target_book', 'recco_book_id', 'similarity_score'])

        def optimized_converter(simbooks, user_id_list, name, prog_bar_description):
            user_ratings_list = pd.DataFrame(columns=['user_id', 'recco_book_id', 'similarity_score'])
            for curr_user_id in tqdm(user_id_list, desc=prog_bar_description):
                curr_user_books = pd.Series(target_users_and_books[curr_user_id])
                relevant_simbooks = simbooks[simbooks['target_book'].isin(curr_user_books)]
                summed_scores = relevant_simbooks.groupby('recco_book_id')['similarity_score'].sum().reset_index()
                summed_scores['user_id'] = curr_user_id
                if not curr_user_books.empty:
                    summed_scores = summed_scores[~summed_scores['recco_book_id'].isin(curr_user_books)]
                    # TODO: Think about how to adjust this for small number of books
                    summed_scores['similarity_score'] /= len(curr_user_books)
                top_30 = summed_scores.nlargest(30, 'similarity_score')
                user_ratings_list = pd.concat([user_ratings_list, top_30], ignore_index=True)
            return user_ratings_list.rename(columns={'recco_book_id': 'book_id', 'similarity_score': name})

        genre_users = optimized_converter(genrebooks, target_user_list, 'gen_score', "Generating recs (genre)")
        cluster_users = optimized_converter(clusterbooks, target_user_list, 'clus_score',
                                            "Generating recs (book stats cluster)")
        description_users = optimized_converter(descbooks, target_user_list, 'desc_score',
                                                "Generating recs (description)")
        reviews_users = optimized_converter(revbooks, target_user_list, 'rev_score', "Generating recs (reviews)")

        #################################################################################
        # USER SIMILARITY CLUSTERING MODEL
        #################################################################################

        def jaccard_similarity_pandas(target_user, reviews_sub, n):
            target_user_books = target_users_and_books[target_user]
            relevant_reviews = reviews_sub[reviews_sub['book_id'].isin(target_user_books)]
            intersections = relevant_reviews.groupby('user_id').size()
            # all_books = pd.concat(
            #     [df[df['user_id'] == target_user]['book_id'], reviews_sub['book_id']]).drop_duplicates()
            user_book_counts = reviews_sub.groupby('user_id')['book_id'].nunique()
            unions = len(target_user_books) + user_book_counts - intersections
            jaccard_index = intersections / unions
            top_n_users = jaccard_index.nlargest(n)
            return top_n_users.reset_index().values.tolist()

        def recommend_books(target_user_id, reviews_sub, num_books):
            # df = reviews_sub[(reviews_sub['rating'].isin([4, 5]))]
            top_n_similar_users = jaccard_similarity_pandas(target_user_id, reviews_sub, n=20)
            target_user_books = target_users_and_books[target_user_id]
            similar_users_reviews = reviews_sub[reviews_sub['user_id'].isin([user[0] for user in top_n_similar_users])]

            recommended_books = defaultdict(float)
            for curr_user_id, similarity_score in top_n_similar_users:
                user_reviews = similar_users_reviews[similar_users_reviews['user_id'] == curr_user_id]
                for _, row in user_reviews.iterrows():
                    if row['book_id'] not in target_user_books:
                        recommended_books[row['book_id']] += similarity_score

            # Return top recommended books sorted by score
            sorted_recommended_books = sorted(recommended_books.items(), key=lambda x: x[1], reverse=True)
            return [(target_user_id, book_id, book_score) for book_id, book_score in
                    sorted_recommended_books[:num_books]]

        all_recommendations = []

        for each_user_id in tqdm(target_user_list, desc="Generating recs (users)"):
            recommendations = recommend_books(each_user_id, file_dict['reviews_sub'], 30)
            all_recommendations.extend(recommendations)
        user_users = pd.DataFrame(all_recommendations, columns=['user_id', 'book_id', 'user_score'])
        user_users.head()

        #################################################################################
        # TITLE SIMILARITY MODEL
        #################################################################################

        store = FAISS.load_local(
            "Data/faiss_store",
            HuggingFaceBgeEmbeddings(
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            ),
            allow_dangerous_deserialization=True
        )

        title_output = []
        for user_id, books in tqdm(target_users_and_books.items(), desc="Generating recs (title)"):
            user_book_id = target_users_and_books[user_id]
            user_books = file_dict['books'][(file_dict['books']['book_id'].isin(user_book_id))]
            titles = '\n'.join(user_books['title_without_series'])  # Using titles without series for queries
            results = store.similarity_search_with_score(titles, k=80)
            for result, score in results:
                if result.metadata.get('book_id') not in user_books:
                    title_output.append([user_id, result.metadata.get('book_id'), 1 - score])

        # Save formatted
        title_users = pd.DataFrame(title_output, columns=['user_id', 'book_id', 'tit_score'])

        #################################################################################
        # COMBINING MODEL OUTPUTS
        #################################################################################

        self.temp_store = {
            'cluster': cluster_users,
            'genre': genre_users,
            'desc': description_users,
            'reviews': reviews_users,
            'users': user_users,
            'title': title_users,
        }

        combined_df = pd.merge(cluster_users, genre_users, on=['user_id', 'book_id'], how='outer')
        combined_df = pd.merge(combined_df, description_users, on=['user_id', 'book_id'], how='outer')
        combined_df = pd.merge(combined_df, reviews_users, on=['user_id', 'book_id'], how='outer')
        combined_df = pd.merge(combined_df, user_users, on=['user_id', 'book_id'], how='outer')
        combined_df = pd.merge(combined_df, title_users, on=['user_id', 'book_id'], how='outer')

        combined_df.fillna(0, inplace=True)
        combined_df['book_id'] = combined_df['book_id'].astype(int)
        combined_df['tit_score'] = combined_df['tit_score'].astype(float)

        reviews_df = file_dict['reviews'][file_dict['reviews']['rating'].isin([1, 2, 3, 4, 5])]
        reviews_filtered = reviews_df[['user_id', 'book_id', 'rating']]
        combined_df = combined_df.merge(reviews_filtered, on=['user_id', 'book_id'], how='left')
        combined_df.rename(columns={'rating': 'target'}, inplace=True)
        combined_df['binary'] = np.where(combined_df['target'] >= 4, 1, 0)

        # remove books which are not recommended at all
        combined_df = combined_df[
            (combined_df[['clus_score', 'gen_score', 'desc_score', 'rev_score', 'user_score', 'tit_score']] != 0).any(
                axis=1)]

        with open("Data/final_model.pkl", 'rb') as file:
            self.pipeline = pickle.load(file)

        X_test = combined_df.drop(columns=['user_id', 'book_id', 'target', 'binary'])
        predictions_df = combined_df[
            ['user_id', 'book_id', 'clus_score', 'gen_score', 'desc_score', 'rev_score', 'user_score',
             'tit_score', 'target', 'binary']].copy()
        predictions_df['final_score'] = self.pipeline.predict_proba(X_test).T[1]
        predictions_df['would_recommend'] = predictions_df['final_score'] >= 0.45  # peak f2 score at this threshold
        predictions_df = predictions_df.sort_values(['user_id', 'final_score'], ascending=[True, False])

        self.dataset = combined_df
        self.predictions = predictions_df

    def prepare_user_details(self):
        users_list = self.dataset['user_id'].unique()

        users_df = pd.read_csv("Data/final_dataset/users_test.csv")
        books_df = pd.read_csv("Data/final_dataset/books_test.csv")

        # filter to keep only relevant users
        users_df = users_df[users_df['user_id'].isin(users_list)]
        # merge to get book and review data
        full_df = users_df.merge(books_df, on="user_id")

        user_details = pd.DataFrame()
        top_books_per_user = full_df.groupby("user_id").apply(
            lambda x: x.sort_values('rating').nlargest(n=5, columns='rating')['title_without_series'].tolist())
        user_details['top_books'] = top_books_per_user

        self.user_details = user_details

    def get_user_predictions(self, chosen_user):
        logging.info(f"Generating predictions for user: {chosen_user}")
        user_predictions = self.predictions[self.predictions['user_id'] == chosen_user]
        user_predictions = user_predictions.dropna(subset=['target'])
        if len(user_predictions) == 0:
            logging.info(f"No predictions hit! Exiting early")
            return None

        # transform model scores using the pipeline (scaler + logistic regression coefficients)
        # specifically, apply scaler then apply linear layer of logistic regression
        model_score_cols = [c for c in user_predictions.columns if c.endswith('_score') and c != 'final_score']
        scaled_model_scores = self.pipeline['scaler'].transform(user_predictions[model_score_cols])
        multed_model_scores = scaled_model_scores * self.pipeline['classifier'].coef_[0]
        final_model_scores = pd.DataFrame(multed_model_scores, columns=model_score_cols)
        final_model_scores['intercept'] = self.pipeline['classifier'].intercept_[0]

        columns = ['book_id', 'target', 'final_score', 'would_recommend']
        predictions_and_score = pd.concat(
            [user_predictions[columns].reset_index(drop=True), final_model_scores],
            axis=1
        )
        return predictions_and_score.merge(self.all_books[['book_id', 'title_without_series']], on='book_id')

    def cache_results(self):
        with open(self.cache_path, 'wb+') as f:
            to_pickle = dict()
            to_pickle['dataset'] = self.dataset
            to_pickle['predictions'] = self.predictions
            to_pickle['temp_store'] = self.temp_store
            to_pickle['pipeline'] = self.pipeline
            to_pickle['chosen_books'] = self.chosen_books_per_user
            # to_pickle['user_details'] = self.user_details
            pickle.dump(to_pickle, f)
        self.is_loaded = True

    def does_cache_exist(self):
        return os.path.exists(self.cache_path)

    def retrieve_cache(self):
        with open(self.cache_path, 'rb') as f:
            unpickled = pickle.load(f)
            for key, val in unpickled.items():
                exec(f"self.{key} = val")
        self.is_loaded = True
