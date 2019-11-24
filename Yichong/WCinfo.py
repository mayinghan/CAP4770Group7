import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv", index_col=0)

sorted = df.sort_values(by='target', ascending=False)
totalComment = len(sorted)
toxicContains = sorted[(sorted['target'] >= 0.5) & (sorted['severe_toxicity'] > 0) & (sorted['obscene'] > 0) & (sorted['identity_attack'] > 0) & (sorted['insult'] > 0) & (sorted['threat'] > 0)].groupby("target")

columnsData = df.loc[ : , 'comment_text' ]

alist = columnsData.tolist()

def listToString(s):  
    str1 = ""  
    for ele in s:  
        str1 += ele    
    return str1  

ss = listToString(alist)

wordcloud = WordCloud().generate(ss)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


