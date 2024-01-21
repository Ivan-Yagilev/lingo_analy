import spacy
from models import Output


def use_model(input_data):
    # Загружаем сохраненную модель
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    # Определяем возвращаемое предсказание
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "negative"
        score = parsed_text.cats["neg"]
    
    output = Output(prediction=prediction,
                    score=score)
    return output