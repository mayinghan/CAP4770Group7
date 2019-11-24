import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv", index_col=0)

sorted = df.sort_values(by='target', ascending=False)
totalComment = len(sorted)
toxic = sorted[sorted['target'] >= 0.5].groupby("target")

totalToxic = len(toxic.count())

print(totalComment)
# plt.figure(figsize=(15,10))
# toxic.size().sort_values(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("ID")
# plt.ylabel("Toxic")
# plt.show()

labels = 'Toxic', 'non-Toxic'
ratio = totalToxic/totalComment
sizes = [ratio, 1 - ratio]
explode = (0.1,0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()