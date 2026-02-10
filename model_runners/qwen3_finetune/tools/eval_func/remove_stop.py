from nltk.corpus import stopwords
import nltk
import re

cachedStopWords = stopwords.words("english")

#this function removes stopwords and punctuations
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(t for t in text.split() if t not in cachedStopWords)

if __name__ == "__main__":
    dummy_text = ", i want, to test this sentence"
    print(normalize_text(dummy_text))

