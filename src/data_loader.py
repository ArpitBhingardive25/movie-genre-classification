import pandas as pd

def load_train_data(path):
    ids, titles, genres, descriptions = [], [], [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")

            if len(parts) >= 4:
                ids.append(parts[0])
                titles.append(parts[1])
                genres.append(parts[2])
                descriptions.append(" ::: ".join(parts[3:]))

    return pd.DataFrame({
        "id": ids,
        "title": titles,
        "genre": genres,
        "description": descriptions
    })


def load_test_data(path):
    ids, titles, descriptions = [], [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")

            if len(parts) >= 3:
                ids.append(parts[0])
                titles.append(parts[1])
                descriptions.append(" ::: ".join(parts[2:]))

    return pd.DataFrame({
        "id": ids,
        "title": titles,
        "description": descriptions
    })
