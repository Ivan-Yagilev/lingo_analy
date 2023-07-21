import spacy


def use_model(input_data: str):
    # Загружаем сохраненную модель
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    # Определяем возвращаемое предсказание
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный текст"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Негативный текст"
        score = parsed_text.cats["neg"]
    return f"Предсказание: {prediction}\nС вероятностью: {score:.3f}"
