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

myList = df.values.tolist()

str1 = ' '.join(str(v) for v in myList)

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1  


ss = listToString(str1)

wordcloud = WordCloud().generate(ss)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


