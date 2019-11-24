import numpy as np
import pandas as pd
from os import path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py
df = pd.read_csv("new_data.csv", index_col=0)

def plots(features, title):
    plt.title(title)
    for feature in features:
        sns.distplot(df.loc[~df[feature].isnull(),feature],kde=True,hist=False, bins=20, label=feature)
    plt.xlabel('')
    plt.legend()
    plt.show()

features = ['severe_toxicity', 'obscene','identity_attack','insult','threat']
plots(features, "Distribution of additional toxicity features in the train set")