#!/usr/bin/env python
# coding: utf-8

# In[1]:


from confluent_kafka import Consumer, KafkaError
import pandas as pd


# In[2]:


# Kafka bootstrap servers
bootstrap_servers = 'localhost:9092'

# Create Kafka consumer configuration
conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'my_consumer_group', 'auto.offset.reset': 'earliest'}

# Kafka topic to consume messages from
topic = 'project'

# Create Kafka consumer instance
consumer = Consumer(conf)

# Subscribe to the Kafka topic
consumer.subscribe([topic])

# List to store consumed messages
tweets = []

n1=100
n=0


# In[3]:


# Consume messages
while n<n1:
    msg = consumer.poll(1.0)  # 1-second timeout
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            # End of partition event
            continue
        else:
            print(f"Error: {msg.error()}")
            break

    # Process the received message
    tweets.append(msg.value().decode('utf-8'))

    # Break after receiving one message
    n+=1

# Close the Kafka consumer
consumer.close()



# In[4]:


# Convert messages list to DataFrame
df = pd.DataFrame(tweets, columns=['Tweets'])

# Display the DataFrame
print(df)


# In[ ]:




