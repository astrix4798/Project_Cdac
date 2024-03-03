#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


import numpy as np


# In[31]:


import re


# In[32]:


import seaborn as sns


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


from matplotlib import style


# In[35]:


style.use('ggplot')


# In[36]:


from textblob import TextBlob


# In[37]:


from nltk.tokenize import word_tokenize


# In[38]:


from nltk.stem import PorterStemmer


# In[39]:


from nltk.corpus import stopwords


# In[40]:


import nltk


# In[41]:


nltk.download('stopwords')


# In[42]:


stop_words = set(stopwords.words('english'))


# In[43]:


from wordcloud import WordCloud


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


from sklearn.linear_model import LogisticRegression


# In[47]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[48]:


df = pd.read_csv("News_Tweets.csv", index_col=False)


# In[49]:


df


# In[50]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[52]:


import nltk
nltk.download('punkt')


# In[53]:


new_df = pd.DataFrame()
new_df["Tweets"] = df["Tweets"].apply(data_processing)


# In[54]:


new_df


# In[55]:


new_df = new_df.drop_duplicates("Tweets")


# In[56]:


new_df


# In[57]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[58]:


new_df["Tweets"] = new_df["Tweets"].apply(lambda x: stemming(x))


# In[59]:


new_df


# In[60]:


new_df.info()


# In[61]:


def polarity(text):
    return TextBlob(text).sentiment.polarity


# In[62]:


new_df["Polarity"] = new_df["Tweets"].apply(polarity)


# In[63]:


new_df


# In[64]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[65]:


new_df["Sentiment"] = new_df["Polarity"].apply(sentiment)


# In[66]:


new_df


# In[ ]:





# In[68]:


fig = plt.figure(figsize=(8,6))
ax = sns.countplot(x='Sentiment', data=new_df)

# Add count labels to each bar
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')

plt.show()


# In[69]:


fig = plt.figure(figsize=(8,8))
colors = ("green", "blue", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = new_df['Sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')


# In[70]:


new_df['Sentiment'].unique()


# In[71]:


pos_tweets = new_df[new_df['Sentiment'] == 'Positive']
pos_tweets = pos_tweets.sort_values(['Polarity'], ascending= False)
pos_tweets.head()


# In[72]:


nut_tweets = new_df[new_df['Sentiment'] == 'Neutral']
nut_tweets = nut_tweets.sort_values(['Polarity'], ascending= False)
nut_tweets.head()


# In[73]:


neg_tweets = new_df[new_df['Sentiment'] == 'Negative']
neg_tweets = neg_tweets.sort_values(['Polarity'], ascending= False)
neg_tweets.head()


# In[74]:


vect = CountVectorizer(ngram_range=(1,2)).fit(new_df["Tweets"])


# In[75]:


vect


# In[ ]:




