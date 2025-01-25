# Emoji project

## Links

- dataset: https://github.com/datasets/emojis
- stemming nltk: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
- Twitter emoji dataset: https://www.kaggle.com/datasets/hariharasudhanas/twitter-emoji-prediction?select=Train.csv

# Simple solution:

1. Text pre-processing - stemming
2. Mapping from words to emoji words - search in emojis
3. Output

# Middle solution (added embeddings)

1. Text pre-processing - stemming
2. Embed stemmed sentence
3. Get embeddings for emojis
4. Calculate similarity between each word and emojis

The project goal: I have to poo -> 🙋🚽💩

# Harder solution: (maybe try doing after we have previous)

- Note: Have original message mixed with emojis that are generated (hard) 
Something similar to https://github.com/gghati/Emojifier

- Another LSTM example: https://github.com/niladridutt/Text_to_Emoji

# We don't have a life solution and nothing better to do:

https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji

