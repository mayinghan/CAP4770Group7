import numpy as np
import pandas as pd
from os import path
from PIL import Image
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv("new_data.csv", index_col=0)

train, test = train_test_split(df, test_size=0.33, random_state=42)

columnsData = df.loc[ : , 'comment_text' ]
columnsDataTrain = train.loc[ : , 'comment_text' ]
columnsDataTest = test.loc[ : , 'comment_text' ]

#Total column Data
yo = columnsData.unique()
setyo = set(yo)
mylist = list(setyo)
str1 = ''.join(mylist)
tokenizer = RegexpTokenizer(r'\w+')

#Length column Data
wordLength = len(tokenizer.tokenize(str1))
wordList = tokenizer.tokenize(str1)
unique = set(wordList)
uniqueWordList = list(unique) 
uniqueLength = len(uniqueWordList)

#Total Train Data
yo1 = columnsDataTrain.unique()
setyo1 = set(yo1)
mylist1 = list(setyo1)
str2 = ''.join(mylist1)
tokenizer1 = RegexpTokenizer(r'\w+')

#Length Train Data
wordLength1 = len(tokenizer1.tokenize(str2))
wordList1 = tokenizer1.tokenize(str2)
unique1 = set(wordList1)
uniqueWordList1 = list(unique1) 
uniqueLength1 = len(uniqueWordList1)

#Total Test Data
yo2 = columnsDataTest.unique()
setyo2 = set(yo2)
mylist2 = list(setyo2)
str3 = ''.join(mylist2)
tokenizer2 = RegexpTokenizer(r'\w+')

#Length Test Data
wordLength2 = len(tokenizer2.tokenize(str3))
wordList2 = tokenizer2.tokenize(str3)
unique2 = set(wordList2)
uniqueWordList2 = list(unique2) 
uniqueLength2 = len(uniqueWordList2)



print("Total words: {}".format(wordLength))
print ("Total Training data: ", wordLength1)
print ("Total Testing data: ", wordLength2)

def listIsEmpty(x):
    if not x:
        return 0
    else:
        return 1

def checkWord():
    x = 0
    z = 0
    while (x < uniqueLength):
        if listIsEmpty(listIsEmpty(wn.synsets(uniqueWordList[x]))):
            z = z + 1
            x = x + 1
        else:
            x = x + 1
    return z

def checkWordTrain():
    x = 0
    z = 0
    while (x < uniqueLength1):
        if listIsEmpty(listIsEmpty(wn.synsets(uniqueWordList1[x]))):
            z = z + 1
            x = x + 1
        else:
            x = x + 1
    return z

def checkWordTest():
    x = 0
    z = 0
    while (x < uniqueLength2):
        if listIsEmpty(listIsEmpty(wn.synsets(uniqueWordList2[x]))):
            z = z + 1
            x = x + 1
        else:
            x = x + 1
    return z
    
print("Total unknown words: {}".format(checkWord()))
print("Total known words: {}".format(wordLength - checkWord()))

print("Train unknown words: {}".format(checkWordTrain()))
print("Train known words: {}".format(wordLength1 - checkWordTrain()))

print("Test unknown words: {}".format(checkWordTest()))
print("Test known words: {}".format(wordLength2 - checkWordTest()))


# print (listIsEmpty(wn.synsets(tokenizer.tokenize(str1)[0])))
