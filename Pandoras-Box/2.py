from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

sentences = [
    "I love this product, it's amazing!",
    "This is the worst service I've ever had.",
    "I'm so happy with my purchase, highly recommend!",
    "I'm not satisfied at all with this experience."
]

results = sentiment_pipeline(sentences)

print(f"{'Sentence':<50} | {'Label':<10} | {'Score'}")
print("-" * 75)

for sentence, result in zip(sentences, results):
    label = result['label']
    score = round(result['score'], 4)
    print(f"{sentence:<50} | {label:<10} | {score}")
