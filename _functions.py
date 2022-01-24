# Create the Bitcoin sentiment scores DataFrame
# Initial imports
import os
try:
    width = os.get_terminal_size().columns
except OSError:
    pass 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from newsapi import NewsApiClient
import pandas as pd
from dotenv import load_dotenv
from nltk.corpus import stopwords, reuters
import nltk as nltk
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
from nltk import ngrams
from string import punctuation
lemmatizer = WordNetLemmatizer()
import re
tokenization_df = pd.DataFrame()
from wordcloud import WordCloud

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()
# Create a list of stopwords
nltk_stop_words = set(stopwords.words('english'))
stop_words_list = list(nltk_stop_words)
# Expand the default stopwords list if necessary
include_in_stopwords = ['reuters']
remove_from_stopwords = ['all', 'below', 'being', 'too']
stop_words = [elem for elem in stop_words_list if elem not in remove_from_stopwords]



def get_sentiment(score):
    """
    Calculates the sentiment based on the compound score.
    """
    result = 0  # Neutral by default
    if score >= 0.05:  # Positive
        result = 1
    elif score <= -0.05:  # Negative
        result = -1
    return result


# Create the Bitcoin sentiment scores DataFrame
def create_btcusd_df(bitcoin_news, language):
    btcusd_articles = []
    for article in bitcoin_news:
        try:
            title = article["title"]
            description = article["description"]
            text = article["content"]
            date = article["publishedAt"][:10]

            btcusd_articles.append({
                "title": title,
                "description": description,
                "text": text,
                "date": date,
                "language": language
            })
        except AttributeError:
            pass

    return pd.DataFrame(btcusd_articles)



# Create the Ethereum sentiment scores DataFrame
def create_ethusd_df(ethereum_news, language):
    ethusd_articles = []
    for article in ethereum_news:
        try:
            title = article["title"]
            description = article["description"]
            text = article["content"]
            date = article["publishedAt"][:10]

            ethusd_articles.append({
                "title": title,
                "description": description,
                "text": text,
                "date": date,
                "language": language
            })
        except AttributeError:
            pass

    return pd.DataFrame(ethusd_articles)


    # Complete the tokenizer function
def process_text(article):
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', article)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    output = [word.lower() for word in lem if word.lower() not in stop_words]
    return output


def process_text_grams(tokenized):
    bigrams = ngrams(tokenized, 2)
    output = ['_'.join(i) for i in bigrams]
    return ' '.join(output)



    # Function token_count generates the top 10 words for a given coin
def word_count(article):
    big_string = ' '.join(article)
    processed = process_text(big_string)
    top_10 = dict(Counter(processed).most_common(10))
    return pd.DataFrame(list(top_10.items()), columns=['word', 'count'])
    """Returns the top N tokens from the frequency count"""
    return Counter(tokens).most_common(N)


    
def process_text_bg(doc):
    sw = stop_words
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', doc)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    sw_words = [word.lower() for word in lem if word.lower() not in stop_words]
    bigrams = ngrams(sw_words, 2)
    output = ['_'.join(i) for i in bigrams]
    return ' '.join(output)

    
def tokenizer(text):
    #    """Tokenizes text."""
    sw = stop_words
    # Remove the punctuation from text
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)
    # Create a tokenized list of the words
    words = word_tokenize(re_clean)
    # Lemmatize words into root words
    lem = [lemmatizer.lemmatize(word) for word in words]
    # Remove the stop words
    # Convert the words to lowercase
    sw_words = [word.lower() for word in lem if word.lower() not in stop_words]
    bigrams = ngrams(sw_words, 2)
    output = ['_'.join(i) for i in bigrams]
    return ' '.join(output)
