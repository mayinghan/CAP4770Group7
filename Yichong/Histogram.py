import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv", index_col=0)

sorted = df.sort_values(by='target', ascending=False)
totalComment = len(sorted)
severeToxic = sorted[sorted['severe_toxicity'] > 0].groupby("severe_toxicity")
obscene = sorted[sorted['obscene'] > 0].groupby("obscene")
identity_attack = sorted[sorted['identity_attack'] > 0].groupby("identity_attack")
insult = sorted[sorted['insult'] > 0].groupby("insult")
threat = sorted[sorted['threat'] > 0].groupby("threat")

severeToxicCount = len(severeToxic.count())
obsceneCount = len(obscene.count())
identity_attackCount = len(identity_attack.count())
insultCount = len(insult.count())
threatCount = len(threat.count())

x = np.arange(5)
plt.bar(x, height= [severeToxicCount, obsceneCount, identity_attackCount, insultCount, threatCount])
plt.xticks(x, ['SevereToxic','Obscene','Identity_attack','Insult', 'Threat'])
plt.xlabel("Category")
plt.ylabel("Counts")
plt.title("Category vs Counts")
plt.show()