import numpy as np
import pandas as pd
from os import path
from PIL import Image
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv", index_col=0)

#https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
plt.figure(figsize=(12,6))
plt.title("Distribution of target in new data set")
sns.distplot(df['target'],kde=True,hist=False, bins=20, label='target')
plt.legend(); 
plt.show()