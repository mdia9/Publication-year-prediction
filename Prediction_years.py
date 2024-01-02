#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages

import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# importing necessary libraries

import numpy as np
import re
# nltk.stopwords
from nltk.corpus import stopwords

# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# functions

# string cleaner function
def string_cleaner(colum):
    # lowercase
    colum = colum.str.lower()

    # removing stopwords from the data
    stop_words = stopwords.words("english")
    colum = colum.apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

    # remove special characters
    colum = colum.str.replace("\W", ' ', regex=True) # https://www.statology.org/pandas-remove-special-characters/

    # Apply lemmatization
    wnl = WordNetLemmatizer()
    colum = colum.apply(lambda x: " ".join(wnl.lemmatize(word, "v") for word in x.split()))

    return colum

# function to calculate the number of authors
def author_count(column):
    n_authors = []
    for i in column:
        count = 0
        for j in i:
            count += 1
        n_authors.append(count)
    
    return n_authors

# function for publisher column
def trans_publisher(column):
    has_publisher = []
    for x in column:
        if len(x) < 1:
            has_publisher.append(False)
        else:
            has_publisher.append(True)
    
    return has_publisher

# function for editor column
def trans_editor(column):
    has_editor = []
    for x in column:
        if len(x) < 1:
            has_editor.append(False)
        else:
            has_editor.append(True)
    
    return has_editor

# function to transform the abstract
def trans_abstract(column):
    has_abstract = []
    for x in column:
        if len(x) < 1:
            has_abstract.append(False)
        else:
            has_abstract.append(True)
    
    return has_abstract

# function for cleaning author column
def author_cleaning(column):
    column = column.astype(str)
    for x in range(0, len(column)):
        column[x] = column[x].strip("[]")
    
    return column


def main():

    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")

    # load the data
    train = pd.DataFrame.from_records(json.load(open("train.json"))).fillna("")
    test = pd.DataFrame.from_records(json.load(open("test.json"))).fillna("")

    # drop the editor column
    train = train.drop(["editor"], axis= 1)
    test = test.drop(["editor"], axis= 1)

    # creating and adding to database the column number of authors
    train["n_author"] = author_count(train["author"])
    test["n_author"] = author_count(test["author"])

    # trasforming the author column into a string and clean it
    train["author"] = author_cleaning(train["author"])
    test["author"] = author_cleaning(test["author"])

    # has abstract column
    train["has_abstract"] = trans_abstract(train["abstract"])
    test["has_abstract"] = trans_abstract(test["abstract"])

    # has publisher column
    train["has_publisher"] = trans_publisher(train["publisher"])
    test["has_publisher"] = trans_publisher(test["publisher"])

    # cleaning the title column
    train["title"] = string_cleaner(train["title"])
    test["title"] = string_cleaner(test["title"])

    # cleaning the abstract column
    train["abstract"] = string_cleaner(train["abstract"])
    test["abstract"] = string_cleaner(test["abstract"])
     # cleaning publisher column
    train["publisher"] = string_cleaner(train["publisher"])
    test["publisher"] = string_cleaner(test["publisher"])

    # split the dataset
    train, val = train_test_split(train, stratify= train["year"], test_size= 0.2, random_state= 123)

    # featurizer
    featurizer = ColumnTransformer(transformers=[("title", CountVectorizer(), "title"), 
                                                    ("ENTRYTYPE", OneHotEncoder(handle_unknown= "ignore"), ["ENTRYTYPE"]),
                                                    ("publisher", CountVectorizer(), "publisher"),
                                                    ("author", CountVectorizer(), "author"),
                                                    ("abstract", CountVectorizer(), "abstract"),
                                                    ("n_author", "passthrough", ["n_author"]),
                                                    ("has_abstract", "passthrough", ["has_abstract"]),
                                                    ("has_publisher", "passthrough", ["has_publisher"])], 
                                                    remainder='drop')



    # make a pipeline for the models
    rf_reg = make_pipeline(featurizer, RandomForestRegressor(n_estimators= 300, min_samples_split= 5, criterion = "squared_error", n_jobs= -1, random_state= 123))

    # fit the models
    rf_reg.fit(train.drop("year", axis= 1), train["year"].values)

    logging.info("Evaluating on validation data")

    # calculate the error
    err = mean_absolute_error(val['year'].values, np.round(rf_reg.predict(val.drop('year', axis=1))))
    logging.info(f"random forest regressor MAE: {err}")
    #pred = rf_reg.predict(test)
    pred = rf_reg.predict(test)
    test['year'] = np.round(pred)
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)

main()


# In[ ]:




