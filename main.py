import polars as pl
from polars.datatypes import Struct
from collections import Counter


train_data = pl.read_parquet('data/otto-reduced/train.parquet')
pl.Config.set_fmt_str_lengths(1000)


class Recommender:
    def fit(self, data: pl.DataFrame) -> None:
        # fit the model to the training data
        pass

    def recommend(self, events: list[Struct]) -> list[int]:
        # return a list of k item ids
        pass


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

recommender_baseline = BaselineRecommender()
recommender_baseline.fit(train_data)
print("trained")
del train_data
test_data = pl.read_parquet('data/otto-reduced/test.parquet')
evaluation(recommender, test_data)
evaluation(recommender_baseline, test_data)
# print(recommendations)
