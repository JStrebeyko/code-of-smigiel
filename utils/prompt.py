def get_abstract_prompt(prefix):
    return f"""Napisz po polsku abstrakt dla tekstu naukowego zatytułowanego {prefix}. Nie powtarzaj tutułu artykułu. Zwróć wyłącznie treść abstraktu, nic poza nim."""

def get_coursebook_prompt(text):
    return f"""Napisz po polsku ustęp do podręcznika szkolnego zatytułowanego {text}. Nie powtarzaj tytułu podręcznika, nazwy rozdziału ani tematu. Zwróć jedynie treść ustępu, nic poza nim."""

def get_classic_lit_prompt(text):
    prompt = f"Dopisz po polsku dalszy ciąg do podanego fragmentu utworu literackiego. Jeśli pochodzi z istniejącego już dzieła, upewnij się, że go nie cytujesz. Zwróć wyłącznie tekst Twojego autorstwa, nic poza nim. \"{text}\"."
    return prompt

def get_wiki_prompt(prefix):
    return f"""Zachowując styl encyklopedyczny, wygeneruj ciąg dalszy artykułu, który zaczyna się od "{prefix}". """

def get_gov_prompt(prefix):
   return f"""Wypowiedź polskiego parlamentarzysty lub polskiej parlamentarzystki zaczyna się od: “{prefix}”. Zachowując styl, dopisz dalszy ciąg tej wypowiedzi. Nie formatuj wypowiedzi w paragrafy i nie umieszaj jej w cudzysłowie."""

def get_wikinews_prompt(prefix):
  return f"""Zachowując jego styl, dopisz ciąg dalszy artykułu prasowego, zaczynającego się od “{prefix}”. Nie dodawaj komentarzy. Nie dodawaj pustych linii między paragrafami."""

def get_social_prompt(social_string):
    template_for_review_prompt = f"""Dokończ następujący post, zachowując jego styl i język. Zwróć wyłącznie wygenerowaną treść. "{social_string}"."""
    return template_for_review_prompt


name_map = {
    'hotels': 'hotelu',
    'medicine': 'lekarza',
    'products': 'produktu',
    'courses': 'kursu na uniwersytecie',
    'movies': 'filmu',
}

def get_review_prompt(prefix, category):
    template_for_review_prompt = f"""Zachowaj styl i język recenzji {name_map[category]} i wygeneruj kontunuację zaczynającą się od podanego niżej zdania. Zwróć wyłącznie wygenerowany fragment tekstu. "{prefix}"."""
    return template_for_review_prompt

