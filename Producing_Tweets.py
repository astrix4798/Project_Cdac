#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from ntscraper import Nitter


# In[7]:


scraper=Nitter(0)


# In[12]:


tweets = scraper.get_tweets('the_hindu', mode = 'user', number=100)


# In[13]:


tweets


# In[14]:


from confluent_kafka import Producer  # Import the Producer class

# Kafka bootstrap servers
bootstrap_servers = 'localhost:9092'

# Create Kafka producer configuration
conf = {'bootstrap.servers': bootstrap_servers}

# Create Kafka producer instance
producer = Producer(conf)

topic = 'project'


# In[16]:


for tweet in tweets['tweets']:
       producer.produce(topic, value=tweet['text'].encode('utf-8'))

producer.flush()


# In[17]:


tweets_data = []
for tweet in tweets['tweets']:
        tweets_data.append(tweet['text'])



df = pd.DataFrame(tweets_data, columns=['Tweets'])

# Display the DataFrame
print(df)


# In[18]:


df.to_csv("News_Tweets.csv", index=False)


# In[ ]:





# In[ ]:




