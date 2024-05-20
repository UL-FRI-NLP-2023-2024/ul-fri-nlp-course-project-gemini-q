import json

TEST_TRAIN_SPLIT = 0.9


def preprocess_data(data):
    data_new = []
    for entry in data:
        entry_new = {}
        entry_new["input"] = entry["question"]
        entry_new["output"] = entry["answer"]
        data_new.append(entry_new)
    return data_new


if __name__ == "__main__":
    with open("combine.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    data = preprocess_data(data)

    length = len(data)
    split = int(length * TEST_TRAIN_SPLIT)
    train_data = data[:split]
    test_data = data[split:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open("test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
