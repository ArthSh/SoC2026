"""
Week 4 - Sentiment Analysis using Transformers
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

sentences = [

    "The inspection system detected defects accurately.",

    "The production line experienced repeated quality failures.",

    "The AI model achieved excellent performance during testing."

]

for sentence in sentences:

    result = classifier(sentence)[0]

    print("-"*50)
    print("Sentence :", sentence)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"],4))
