
# CAP4770Group7
CAP4770 Final Project
'Jigsaw Unintended Bias in Toxicity Classification' on Kaggle
Kaggle Dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

Members: Yinghan Ma, Nicholas Miller, Yichong Ma, Gregory DeCanio

## File Structure
#### `Gregory`
- #### `TF_IDF Text Classification.ipynb` - Jupiter Notebook where project code is written. Performs data cleaning (stop word and character removal). Uses TF_IDF and simple classification methods
- #### `README.md` - Contains links to websites that were used as reference when coding
#### `Nicholas`
- #### `README.md` - Contains link to *Google Colab* where are project code is written
#### `Yichong`
- #### `__pycache__`
	- #### `wordcloud.cpython-37.pyc`  
- #### `AdditionalTarget.py`
- #### `Histogram.py`
- #### `PieChart.py`  
- #### `README.md` 
- #### `TargetFeature.py`
- #### `WCObscene75.py`  
- #### `WCPopWords.py`
- #### `WCInfo.py`
- #### `unknownWordGraph.py`
- #### `unknownWords.py`    
#### `Yinghan`
- #### `README.md` 
- #### `__init__.py`
- #### `clean_data.py` 
- #### `data_utils.py`
- #### `model.py`
- #### `run.sh`
- #### `task.py`
- #### `weightpath.py`      
#### `.gitignore` - Tells git which files to ignore
#### `README.md` - This file! Describes problem, file structure, and what each group member is doing
#### `new_data.csv` - Sampled data of 115,000 comments from original 1.8+ million
#### `requirement.txt` - 

## Visuals:
### Method 1: Yichong
+ Analysis 
+ Graphs 
+ Stats

### Method 2: Nick 
+ Word2vec 

## Predictors:
### Method 1: Yinghan 
+ Transformer (Bert)

### Method 2: Nick
+ LSTM

### Method 3: Greg 
Performing the following classification methods using TF_IDF.
Output accuracy, precision, recall, F1-Score, and time needed to construct model.
+ Naïve Bayes (NB) 
+ Stochastic Gradient Descent (SGD)
+ Random Forest (RF)
+ Decision Tree (DT)
