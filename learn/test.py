import spacy


def test_model(input_data: str):
    # Загружаем сохраненную модель
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    # Определяем возвращаемое предсказание
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "good"
        score = parsed_text.cats["pos"]
    else:
        prediction = "bad"
        score = parsed_text.cats["neg"]
    return prediction, score


if __name__ == "__main__":
    TEST_REVIEW = "test"
    print(test_model(input_data=TEST_REVIEW))
