import polars as pl
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from polars.datatypes import Struct
from collections import Counter


train_data = pl.read_parquet('data/otto-reduced/train.parquet')
pl.Config.set_fmt_str_lengths(1000)

# train_data = train_data[0:1000]
# smaller_train = smaller_train.explode("events").unnest("events")
# small_df = smaller_train.to_pandas()
test_data = pl.read_parquet('data/otto-reduced/test_10k.parquet')

"""output_dict = {}
i = 0
for row in smaller_train.iter_rows():
    if i % 1000 == 0:
        print(i)
    session_id = row[0]
    session_clicks = row[1]
    output_dict[session_id] = session_clicks
    i += 1"""


class Recommender:
    def fit(self, data: pl.DataFrame) -> None:
        # fit the model to the training data
        pass

    def recommend(self, events: list[Struct]) -> list[int]:
        # return a list of k item ids
        pass


class SessionBasedKNN:
    def __init__(self, k=10, metric="cosine"):
        self.k = k
        self.metric = metric
        self.item_vectors = None
        self.nn_model = None

    def fit(self, session_data):
        # Create item-session matrix (consider only "clicks" for similarity)
        item_session_matrix = defaultdict(set)
        for session, interactions in session_data.items():
            for interaction in interactions:
                if interaction["type"] == "clicks":
                    item_session_matrix[interaction["aid"]].add(session)

        # Transform to DataFrame for easier manipulation
        df = pd.DataFrame.from_dict(item_session_matrix, orient="index") # .fillna(0)
        df = df.astype(bool).astype(int)  # Convert to binary representation

        # Calculate item vectors and fit kNN model
        self.item_vectors = df.values
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn_model.fit(self.item_vectors)

    def predict_next_click(self, current_session):
        # Get item IDs from the current session (consider only "clicks")
        clicked_items = [interaction[0] for interaction in current_session if interaction[2] == "clicks"]

        if not clicked_items:  # Handle empty session case
            return None

        # Average the vectors of clicked items to get the session vector
        session_vector = self.item_vectors[clicked_items].mean(axis=0)

        # Find k nearest neighbors
        _, indices = self.nn_model.kneighbors(session_vector.reshape(1, -1))
        neighbors = [df.index[idx] for idx in indices.flatten()]

        # Filter out already clicked items
        recommendations = [item for item in neighbors if item not in clicked_items]

        return recommendations[:self.k]  # Return top k recommendations


def session_row_to_set(session_row):
    output_set = set()
    for item in session_row:
        output_set.add(item["aid"])
    return output_set


class BaselineRecommender(Recommender):
    def __init__(self):
        self.top_k = None

    def fit(self, data):
        data = data.explode("events").unnest("events")
        self.top_k = data.group_by("aid").len().sort("len", descending=True).head(20)["aid"].to_list()

    def recommend(self, events):
        return self.top_k


def evaluation(model, test_data):
    # evaluate the model on the test data
    right_predictions = 0
    i = 1
    last_perc = 0.05
    test_len = len(test_data)
    print("Test data: ", test_len)
    for sequence in test_data.iter_rows():
        if i / test_len > last_perc:
            print(i)
            last_perc += 0.05
        recommendations = model.recommend(sequence[2])
        label = sequence[1]
        if label in recommendations:
            right_predictions += 1

        if i % 10 == 0:
            print(f"Accuracy so far: {right_predictions / i * 100} %")
        i += 1

    print(f"Accuracy: {right_predictions / len(test_data) * 100} %")


class SessionBasedRecommender(Recommender):
    def __init__(self):
        self.session_item_matrix = {}
        self.top_n = 20
        self.top_k = None

    def fit(self, data):
        i = 0
        print("Data len: ", len(data))
        for session in data.iter_rows():
            if i % 100000 == 0:
                print(i)
            self.session_item_matrix[session[0]] = session_row_to_set(session[1])
            i += 1

        data = data.explode("events").unnest("events")
        self.top_k = data.group_by("aid").len().sort("len", descending=True).head(20)["aid"].to_list()

    def recommend(self, events):
        similar_sessions = []
        events_set = session_row_to_set(events)
        for session_id, items in self.session_item_matrix.items():
            intersection_len = len(items.intersection(events_set))
            if intersection_len:
                similar_sessions.append((session_id, intersection_len))

        clicked_items = []
        for session_id, weight in similar_sessions:
            extend_by = list(set(self.session_item_matrix[session_id]) - events_set)
            clicked_items.extend(extend_by * weight)

        item_counts = Counter(clicked_items)
        predicted_items = [item for item, _ in item_counts.most_common(self.top_n)]
        if len(predicted_items) == 0:
            predicted_items = self.top_k

        return predicted_items


# Example usage
recommender = SessionBasedRecommender()
recommender.fit(train_data)

# recommendations = recommender.recommend(test_data.row(0))
evaluation(recommender, test_data)
# print(recommendations)
