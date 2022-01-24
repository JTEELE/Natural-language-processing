# Initial imports
from _functions import *
import os
try:
    width = os.get_terminal_size().columns
except OSError:
    pass 
import plotly.express as px
import plotly.graph_objects as go
import spacy
from spacy import displacy
from plotly.subplots import make_subplots
from newsapi import NewsApiClient
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from collections import Counter
from nltk import ngrams
lemmatizer = WordNetLemmatizer()
import matplotlib as mpl
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
from string import punctuation
import re
tokenization_df = pd.DataFrame()
from wordcloud import WordCloud
load_dotenv()

# Read your api key environment variable
api_key = os.getenv("NEWS_API_KEY")

# Create a newsapi client
newsapi = NewsApiClient(api_key=api_key)

# Fetch the Bitcoin news articles
bitcoin_news = newsapi.get_everything(
    q="bitcoin",
    language="en"
)

bitcoin_news["totalResults"]
raw_btcusd = pd.DataFrame(bitcoin_news)

# Fetch the Ethereum news articles
ethereum_news = newsapi.get_everything(
    q="ethereum",
    language="en"
)

ethereum_news["totalResults"]
raw_ethusd = pd.DataFrame(ethereum_news)


bitcoin_news_df = create_btcusd_df(bitcoin_news["articles"], "en")
print(f'Stored all news articles into dataframe:')
print(f'{bitcoin_news_df.head(1)}')
print('')

print('Using nltk library to calculate sentiment scores..')
title_sent = {
    "title_compound": [],
    "title_pos": [],
    "title_neu": [],
    "title_neg": [],
    "title_sent": [],
}
text_sent = {
    "text_compound": [],
    "text_pos": [],
    "text_neu": [],
    "text_neg": [],
    "text_sent": [],
}

# Get sentiment for the text and the title
for index, row in bitcoin_news_df.iterrows():
    try:
        # Sentiment scoring with VADER
        title_sentiment = analyzer.polarity_scores(row["title"])
        title_sent["title_compound"].append(title_sentiment["compound"])
        title_sent["title_pos"].append(title_sentiment["pos"])
        title_sent["title_neu"].append(title_sentiment["neu"])
        title_sent["title_neg"].append(title_sentiment["neg"])
        title_sent["title_sent"].append(get_sentiment(title_sentiment["compound"]))

        text_sentiment = analyzer.polarity_scores(row["text"])
        text_sent["text_compound"].append(text_sentiment["compound"])
        text_sent["text_pos"].append(text_sentiment["pos"])
        text_sent["text_neu"].append(text_sentiment["neu"])
        text_sent["text_neg"].append(text_sentiment["neg"])
        text_sent["text_sent"].append(get_sentiment(text_sentiment["compound"]))
    except AttributeError:
        pass

# Attaching sentiment columns to the News DataFrame
title_sentiment_df = pd.DataFrame(title_sent)
text_sentiment_df = pd.DataFrame(text_sent)
bitcoin_news_df = bitcoin_news_df.join(title_sentiment_df).join(text_sentiment_df)
print('')
print(f'Joined sentiment scores to news article dataframe:')

bitcoin_news_df.head(2)


ethereum_news_df = create_ethusd_df(ethereum_news["articles"], "en")
print(f'Stored all news articles into dataframe:')
print(f'{ethereum_news_df.head(2)}')
print('')

print('Using nltk library to calculate sentiment scores..')
# Sentiment scores dictionaries
title_sent = {
    "title_compound": [],
    "title_pos": [],
    "title_neu": [],
    "title_neg": [],
    "title_sent": [],
}
text_sent = {
    "text_compound": [],
    "text_pos": [],
    "text_neu": [],
    "text_neg": [],
    "text_sent": [],
}

# Get sentiment for the text and the title
for index, row in ethereum_news_df.iterrows():
    try:
        # Sentiment scoring with VADER
        title_sentiment = analyzer.polarity_scores(row["title"])
        title_sent["title_compound"].append(title_sentiment["compound"])
        title_sent["title_pos"].append(title_sentiment["pos"])
        title_sent["title_neu"].append(title_sentiment["neu"])
        title_sent["title_neg"].append(title_sentiment["neg"])
        title_sent["title_sent"].append(get_sentiment(title_sentiment["compound"]))

        text_sentiment = analyzer.polarity_scores(row["text"])
        text_sent["text_compound"].append(text_sentiment["compound"])
        text_sent["text_pos"].append(text_sentiment["pos"])
        text_sent["text_neu"].append(text_sentiment["neu"])
        text_sent["text_neg"].append(text_sentiment["neg"])
        text_sent["text_sent"].append(get_sentiment(text_sentiment["compound"]))
    except AttributeError:
        pass

# Attaching sentiment columns to the News DataFrame
title_sentiment_df = pd.DataFrame(title_sent)
text_sentiment_df = pd.DataFrame(text_sent)
ethereum_news_df = ethereum_news_df.join(title_sentiment_df).join(text_sentiment_df)
print('')
print(f'Joined sentiment scores to news article dataframe:')
ethereum_news_df.head(3)

# Describe the Bitcoin Sentiment
bitcoin_sentiment_stats = bitcoin_news_df.describe()
try:
    print("Bitcoin Sentiment Results".center(width))
except:
    print('Bitcoin Sentiment Results')
print('')
print('')


#Print summary of neutral, positive & negative.
print('')
btcusd_negative = bitcoin_sentiment_stats['title_neg'].iloc[1]
btcusd_negative_percentage = '{:.1%}'.format(btcusd_negative)
print(f'{btcusd_negative_percentage} of Bitcoin articles are negative')
print('')

btcusd_positive = bitcoin_sentiment_stats['title_pos'].iloc[1]
btcusd_positive_percentage = '{:.1%}'.format(btcusd_positive)
print(f'{btcusd_positive_percentage} of Bitcoin articles are positive')
print('')

btcusd_neutral = bitcoin_sentiment_stats['title_neu'].iloc[1]
btcusd_neutral_percentage = '{:.1%}'.format(btcusd_neutral)
print(f'{btcusd_neutral_percentage} of Bitcoin articles are neutral')


# Describe the Ethereum Sentiment
ethusd_sentiment_stats = ethereum_news_df.describe()
try:
    print("Ethereum Sentiment Results".center(width))
except:
    print('Ethereum Sentiment Results')
print('')
print(ethusd_sentiment_stats)
print('')

# Visualize title and text sentiment
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=ethereum_news_df.index, y=ethereum_news_df['title_sent'], name="Title Sentiment"),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=ethereum_news_df.index, y=ethereum_news_df['text_sent'], name="Text Sentiment"),
    row=1, col=1, secondary_y=True,
)
fig.show()
print('')
ethusd_negative = ethusd_sentiment_stats['title_neg'].iloc[1]
ethusd_negative_percentage = '{:.1%}'.format(ethusd_negative)
print(f'{ethusd_negative_percentage} of ethereum articles are negative')
print('')
ethusd_positive = ethusd_sentiment_stats['title_pos'].iloc[1]
ethusd_positive_percentage = '{:.1%}'.format(ethusd_positive)
print(f'{ethusd_positive_percentage} of ethereum articles are positive')
print('')
ethusd_neutral = ethusd_sentiment_stats['title_neu'].iloc[1]
ethusd_neutral_percentage = '{:.1%}'.format(ethusd_neutral)
print(f'{ethusd_neutral_percentage} of ethereum articles are neutral')



# Instantiate the lemmatizer

# Create a list of stopwords
nltk_stop_words = set(stopwords.words('english'))
stop_words_list = list(nltk_stop_words)
# Expand the default stopwords list if necessary
include_in_stopwords = ['reuters']
remove_from_stopwords = ['all', 'below', 'being', 'too']
stop_words = [elem for elem in stop_words_list if elem not in remove_from_stopwords]
print(stop_words)


# Create a new tokens column for Bitcoin
btcusd_text = bitcoin_news_df.text.tolist()
btcusd_tokenized = []
for row in btcusd_text:
    tokenization = process_text(row)
    btcusd_tokenized.append(tokenization)
bitcoin_news_df['tokenized'] = btcusd_tokenized
bitcoin_news_df.head(2)



# Create a new tokens column for Ethereum
ethusd_text = ethereum_news_df.text.tolist()
ethusd_tokenized = []
for row in ethusd_text:
    tokenization = process_text(row)
    ethusd_tokenized.append(tokenization)
ethereum_news_df['tokenized'] = ethusd_tokenized
ethereum_news_df.head(2)


# Generate the Bitcoin N-grams where N=2
btcusd_grams = []
for row in btcusd_tokenized:
    grams = process_text_grams(row)
    btcusd_grams.append(tokenization)
print(f'Check: {len(btcusd_grams)} BTCUSD ngrams vs. {len(bitcoin_news_df.index)} BTCUSD raw articles')
print('')
csv = pd.DataFrame(btcusd_grams)



# Generate the Ethereum N-grams where N=2
ethusd_grams = []
for row in ethusd_tokenized:
    grams = process_text_grams(row)
    btcusd_grams.append(tokenization)
print(f'Check: {len(ethusd_grams)} ETHUSD ngrams vs. {len(ethereum_news_df.index)} ETHUSD raw articles')
print('')
# Use token_count to get the top 10 words for Bitcoin
btcusd_top_count = word_count(btcusd_text)
# Use token_count to get the top 10 words for Ethereum
ethusd_top_count = word_count(ethusd_text)
print(ethusd_grams)