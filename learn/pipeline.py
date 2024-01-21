# make pipeline

import spacy
nlp = spacy.load("ru_core_news_sm")

text = "Который год - в стране царствует смута и развал, властвует произвол финансово- чиновничьей олигархии. Который год мы ожидаем обещанного благополучия и процветания, получая взамен безудержный рост цен, неплатежи по зарплатам и социальным пособиям, межнациональные войны и конфликты, бандитизм и коррупцию."

doc = nlp(text)

# токенизация текста и удаление стоп-слов (список можно настроить)
filtered_tokens = [token for token in doc if not token.is_stop]

# нормализация с помощью лемматизации
lemmas = [
    f"Token: {token}, lemma: {token.lemma_}"
    for token in filtered_tokens
]

print(lemmas)

# векторизация
vect = list()
for i in range(len(filtered_tokens)):
    vect.append(filtered_tokens[i].vector)