# Data Cleaning and Analysis for Egyptian Arabic Podcasts

## Overview
This repository documents the process of cleaning, preprocessing, and analyzing Egyptian Arabic podcast transcripts. The objective is to extract insights from the data by exploring sentiment trends, linguistic patterns, and topic-based characteristics.

## Dataset
The dataset consists of raw podcast transcripts stored in text files. The preprocessing steps involve cleaning the text, tokenizing words, removing stop words, and performing sentiment analysis.

---

## Data Cleaning
### 1. Removing Timestamps
Podcast transcripts often include timestamps that need to be removed for a cleaner analysis. The script scans through each file, detects timestamps using a regular expression pattern, and removes them. The cleaned text is stored in a separate folder named `cleaned`.

```python
import os
import re

folder_path = "Dataset"

timestamp_pattern = re.compile(r'\b\d{1,2}:\d{2}(:\d{2})?\b')

for root, dirs, files in os.walk(folder_path):
    if "cleaned" in root.split(os.sep):
        continue

    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            cleaned_text = re.sub(timestamp_pattern, '', text)
            cleaned_text = ' '.join(cleaned_text.split())

            save_path = os.path.join(os.path.dirname(root), "cleaned")
            os.makedirs(save_path, exist_ok=True)

            with open(os.path.join(save_path, filename), 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
```

---

## Data Preprocessing
### 2. Tokenization
Tokenization is performed using `nltk` to split the cleaned text into words.

```python
import nltk
from nltk.tokenize import wordpunct_tokenize

folder_path = "Dataset"
data = {}

for root, dirs, files in os.walk(folder_path):
    if "raw_transcripts" in root.split(os.sep):
        continue
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = wordpunct_tokenize(text)
                data[filename] = {
                    "tokens": tokens,
                    "filteredTokens": [],
                    "fullText": text,
                    "sentiment": None,
                }
```

### 3. Stop Word Removal
A combination of NLTK's Arabic stop words and custom stop words specific to Egyptian Arabic are filtered out.

```python
from nltk.corpus import stopwords
nltk.download("stopwords")

arabic_stopwords = set(stopwords.words("arabic"))
custom_stopwords = {"برضه","بتاع","تقول","إنك","الناس", "هاهاها", "يعني"}  
all_stopwords = arabic_stopwords.union(custom_stopwords)

for filename in data:
    filtered_tokens = [word for word in data[filename]["tokens"] if word not in all_stopwords]
    data[filename]["filteredTokens"].extend(filtered_tokens)
```

---

## Data Analysis
### 4. Word Cloud Visualization
A word cloud is generated for each transcript to visualize the most frequently used words.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

font_path = "NotoKufiArabic-VariableFont_wght.ttf"

for filename in data:
    reshaped_text = arabic_reshaper.reshape(" ".join(data[filename]["filteredTokens"]))
    bidi_text = get_display(reshaped_text)

    wordcloud = WordCloud(
        width=800, height=400, font_path=font_path, background_color='white'
    ).generate(bidi_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(f"wordcloud_{filename}.png")
```

### 5. Sentiment Analysis
Sentiment analysis is performed using `camel_tools` to determine the sentiment of each podcast transcript.

```python
from camel_tools.sentiment import SentimentAnalyzer

sa = SentimentAnalyzer.pretrained()
for filename in data:
    text = data[filename]["fullText"]
    sentiment = sa.predict(text)
    data[filename]["sentiment"] = sentiment
    print(f"{filename}: {sentiment}")
```

---

## Exploratory Data Analysis Ideas
Below are some analysis questions we aim to answer using the processed data:
1. **Most Negative Category:** Identify the category with the most negative sentiment.
2. **Consistently Negative Channels:** Determine if there are channels with consistently negative sentiment.
3. **Topic Duration:** Explore whether certain topics take longer to discuss.
4. **Semantic Similarity Among Podcasts:** Investigate if similar podcasts share common semantics.
5. **Identifying Channels by Speech Style:** Analyze if a podcast channel can be identified by its style of speaking.
6. **Most Common Words per Sentiment:** Find the most frequently used words based on sentiment classification.


