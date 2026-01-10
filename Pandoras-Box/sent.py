from transformers import pipeline

# 1. Initialize the sentiment analysis pipeline
# By default, this uses the distilbert-base-uncased-finetuned-sst-2-english model
sentiment_pipeline = pipeline("sentiment-analysis")

# 2. Define the array of strings
sentences = [
    "I love this product, it's amazing!",
    "This is the worst service I've ever had.",
    "I'm so happy with my purchase, highly recommend!",
    "I'm not satisfied at all with this experience."
]

# 3. Execute the analysis
results = sentiment_pipeline(sentences)

# 4. Display the results
print(f"{'Sentence':<50} | {'Label':<10} | {'Score'}")
print("-" * 75)

for sentence, result in zip(sentences, results):
    label = result['label']
    score = round(result['score'], 4)
    print(f"{sentence:<50} | {label:<10} | {score}")
