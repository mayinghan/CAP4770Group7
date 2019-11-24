import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv", index_col=0)

#set STOPWORDS
stopwords = set(STOPWORDS)

#Reference: https://www.datacamp.com/community/tutorials/wordcloud-python
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

#Popular words in sample 10000
show_wordcloud(df['comment_text'].sample(10000))

#or insult > 0.75 in sample 10000

show_wordcloud(df.loc[df['insult'] > 0.75]['comment_text'].sample(10000))
