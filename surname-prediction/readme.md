## Introduction to Sequence Modeling - Russian and English names

---
### Goal
---

Develop and classifier for Russian vs English surnames.

In this iteration we are going to:
* Compute bigram frequencies for English names.
* Compute bigram frequencies for Russian names.
* Develop a bag of bigrams model for distinguishing English and Russian names.
* Implement Good Turing Discounting Model Smoothing
* Test performance of model using English data.

------



```python
import pandas as pd
from pandas import DataFrame
import numpy as np
import re

import collections
from collections import defaultdict, Counter

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer # tokenize texts/build vocab
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer # tokenizes text and normalizes
```

---
### Let's perform some EDA

---


```python
# read the csv file into data frame.
surname_csv = "data_set/russian_and_english_dev.csv"
surname_df = pd.read_csv(surname_csv, index_col = None, encoding="UTF-8")
```


```python
# rename dev data columns.
surname_df.rename(columns = {'Unnamed: 0':'surname', 'Unnamed: 1':'nationality'}, inplace = True)
```


```python
surname_df = surname_df.dropna()
```

---
### Generate Bigrams
Calculate n_grams and frequencies of names

---


```python
# generate ngrams and frequencies
def generate_ngrams(names):
    n_gram = collections.Counter()
    n_gram_freq = 3
    for c in names:
        n_gram.update(Counter(c[idx : idx + n_gram_freq] for idx in range(len(c) - 1)))
        
    return n_gram
```


```python
# retrieve names for computing ngrams
names = open("data_set/corpus/english_names.txt", "r")
english_names = [x.rstrip() for x in names.readlines()]
english_names = [x.lower() for x in english_names]

names = open("data_set/corpus/russian_names.txt", "r")
russian_names = [x.rstrip() for x in names.readlines()]
russian_names = [x.lower() for x in russian_names]
```


```python
eng_gram = generate_ngrams(english_names)
rus_gram = generate_ngrams(russian_names)
```

### Good Turing Smoothing
__Note:__ The smoothing method we will use is the Good-Turing Discounting Formula. It is perfect for accounting for bigrams that have yet to occur.

Equation: C^* = (c + 1) Nc+1/Nc


```python
def good_turing_smoothing(n_gram):
    dict(n_gram)
    smoothing = {}
    
    result = None
    
    
    for k in n_gram:
        result = (n_gram[k] + 1 / n_gram[k])
        smoothing[k] = result
        
    return smoothing
```


```python
# english metaparameters
eng_meta = good_turing_smoothing(eng_gram)
```


```python
# russian metaparameters
rus_meta = good_turing_smoothing(rus_gram)
```

### Feature Selection


```python
# Creating another column for when surname is English or not.
surname_df['label_eng'] = [1 if x =='English' else 0 for x in surname_df['nationality']]
label_eng = surname_df["label_eng"]
```


```python
# Creating another column for when surname is Russian or not.
surname_df['label_rus'] = [1 if x =='Russian' else 0 for x in surname_df['nationality']]
label_rus = surname_df["label_rus"]
```


```python
surname_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>surname</th>
      <th>nationality</th>
      <th>label_eng</th>
      <th>label_rus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Mokrousov</td>
      <td>Russian</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Nurov</td>
      <td>Russian</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Judovich</td>
      <td>Russian</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Mikhailjants</td>
      <td>Russian</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Jandarbiev</td>
      <td>Russian</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Create a bag of ngrams


```python
surname_list = surname_df['surname'].apply(lambda x: re.sub('[^a-zA-Z]', '', x))
```


```python
# vectorize features - unigrams, bigrams, and trigrams
cv = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1,2), strip_accents="ascii", min_df=0.09, max_df=1.0)
X_freq = cv.fit_transform(surname_list)

# tf_transformer for normalization
tf_transformer = TfidfTransformer(use_idf=False).fit(X_freq)
X = tf_transformer.transform(X_freq)
```

------
## Multiple Linear Regression

------

We will train two (2) models: One for English and the other for Russian surnames!


```python
def metaparameters(X, meta):
    for key in meta:
        return meta[key] * X
```


```python
X_eng = metaparameters(X, eng_meta)
X_rus = metaparameters(X, rus_meta)
```

#### English Surname Model


```python
# split the data to train the model
x_train_eng, x_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, label_eng, test_size=0.20)
```


```python
english_model = LinearRegression()
english_model.fit(x_train_eng, y_train_eng)
```




    LinearRegression()



#### Russian Surname Model


```python
x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, label_rus, test_size=0.20)
```


```python
russian_model = LinearRegression()
russian_model.fit(x_train_rus, y_train_rus)
```




    LinearRegression()



### Test Data and Predictions

#### English


```python
englishness_test = english_model.predict(x_test_eng)
```


```python
# summary of results
from statsmodels.api import OLS
OLS(y_test_eng, englishness_test).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>label_eng</td>    <th>  R-squared (uncentered):</th>      <td>   0.632</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.631</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   446.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 21 Sep 2020</td> <th>  Prob (F-statistic):</th>          <td>2.20e-58</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:12:56</td>     <th>  Log-Likelihood:    </th>          <td> -73.597</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   261</td>      <th>  AIC:               </th>          <td>   149.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   260</td>      <th>  BIC:               </th>          <td>   152.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>    1.0301</td> <td>    0.049</td> <td>   21.134</td> <td> 0.000</td> <td>    0.934</td> <td>    1.126</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.685</td> <th>  Durbin-Watson:     </th> <td>   2.103</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.710</td> <th>  Jarque-Bera (JB):  </th> <td>   0.761</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.119</td> <th>  Prob(JB):          </th> <td>   0.683</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.882</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### Russian


```python
russianess_test = russian_model.predict(x_test_rus)
```


```python
# summary of results
OLS(y_test_rus, russianess_test).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>label_rus</td>    <th>  R-squared (uncentered):</th>      <td>   0.857</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.856</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   1557.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 21 Sep 2020</td> <th>  Prob (F-statistic):</th>          <td>9.17e-112</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:12:57</td>     <th>  Log-Likelihood:    </th>          <td> -69.581</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   261</td>      <th>  AIC:               </th>          <td>   141.2</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   260</td>      <th>  BIC:               </th>          <td>   144.7</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>    0.9971</td> <td>    0.025</td> <td>   39.457</td> <td> 0.000</td> <td>    0.947</td> <td>    1.047</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.651</td> <th>  Durbin-Watson:     </th> <td>   2.066</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.266</td> <th>  Jarque-Bera (JB):  </th> <td>   2.512</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.173</td> <th>  Prob(JB):          </th> <td>   0.285</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.666</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



----
### Observations

----

#### (1) Checking if the following names are Russian or English.


```python
# predicting the following names
names = ["Fergus", "Angus", "Boston", "Austin", "Dankworth", "Denkworth", "Birtwistle", "Birdwhistle"]

reshape_feature = cv.transform(names)
english_res = english_model.predict(reshape_feature)
russian_res = russian_model.predict(reshape_feature)

print(f"English Model Results: \n {english_res} \n")
print(f"Russian Model Results: \n {russian_res}")
```

    English Model Results: 
     [ 0.43321572  0.25252268  0.38993089 -0.16778808  0.17241999  0.47250489
      0.100231    0.07384694] 
    
    Russian Model Results: 
     [0.5078675  0.59411788 0.52490733 0.92214076 0.62885159 0.40034486
     0.69007533 0.70230745]


Note: The english model does not see any of the above names as being of English origin. What's interesting is that it sees most of them more as Russian names.

