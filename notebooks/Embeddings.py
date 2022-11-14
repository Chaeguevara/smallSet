#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')


# ## Utility Functions

# Before we start exploring embeddings lets write a couple of helper functions to run Logistic Regression and calculate evaluation metrics
#
# Since we want to optimize our model for F1-Scores, for all models we'll first predict the probability of the positive class. We'll then use these probabilities to get the Precision-Recall curve and from here we can select a threshold value that has the highest F1-score. To predict the labels we can simply use this threshold value.

# In[2]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn import metrics
sns.set_palette("muted")


def calc_f1(p_and_r):
    p, r = p_and_r
    return (2*p*r)/(p+r)


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics
# Since we are optimizing for F1 score - we will first calculate precision and recall and
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob,y_pred,label_list,name):
    print("*"*5)
    print(name)
    print("roc")
    print(roc_auc_score(y_test,y_test_prob,labels = label_list,multi_class='ovr',average="weighted"))
    print("f1")
    print(metrics.f1_score(y_test,y_pred,labels=label_list,average="weighted"))
    print("acc")
    print(metrics.accuracy_score(y_test,y_pred))


# In[3]:


# Run Simple Log Reg Model and Print metrics
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import random
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
import ast
# Run log reg 10 times and average the result to reduce predction variance
def run_log_reg(train_features, test_features, y_train, y_test, lbl_to_idx, idx_to_lbl,label_list,feature_name, apply=True):
    y_train_idx = [lbl_to_idx[label] for label in y_train]
    y_test_idx = [lbl_to_idx[label] for label in y_test]
    label_idx = [i for i in range(len(label_list))]
    models = [
       MultinomialNB(),
       SVC(probability=True),
       RandomForestClassifier(),
#       XGBClassifier(),

    ]
    model_names=[
        "Naive bayes",
        "SVM",
        "RF",
#        "XGBoost"

    ]
    model_prams=[
        {'alpha' : Continuous(0.01,0.5,distribution="log-uniform")},
        {"kernel" : Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
         "C" : Continuous(0.1,50,distribution="uniform")
        },
        {
            "n_estimators" : Integer(10,100),
            "max_depth" : Integer(5,50),
            "min_samples_split" : Integer(2,11),
            "min_samples_leaf" : Integer(1,11),
            "criterion" : Categorical(["gini", "entropy", "log_loss"]),
            "max_features" : Integer(1,13)
        }
    ]
    non_negs = ["BOW", "TF-IDF"]
    if apply == True:
        print(feature_name)
        computed_params = pd.read_csv(f"../result/all_params/{feature_name}.csv", encoding='utf-8')
        computed_params = computed_params.drop(columns=["Unnamed: 0"])
        compute_dict = computed_params.to_dict('dict')

        result_dict ={}
        for model,name in zip(models,model_names):
            if feature_name not in non_negs and name =="Naive bayes":
                continue
            clf = model
            if name in compute_dict:
                clf = model.set_params(**ast.literal_eval(compute_dict[name][0]))
            clf.fit(train_features,y_train_idx)
            y_test_prob = clf.predict_proba(test_features)
            y_pred = clf.predict(test_features)
            print_model_metrics(y_test_idx, y_test_prob, y_pred, label_idx,name)



    pre_comp_best =[]

    random.seed(1)
    cv = cv = StratifiedKFold(n_splits=2, shuffle=True)

    if apply:
        return
    for model, model_name, model_param in zip(models, model_names, model_prams):
        try:
            evolved_estimator = GASearchCV(estimator=model,
                               cv=cv,
                               scoring='accuracy',
                               population_size=10,
                               generations=35,
                               param_grid=model_param,
                               n_jobs=-1,
                               verbose=True,
                               keep_top_k=4)
            evolved_estimator.fit(train_features,y_train_idx)
            print(evolved_estimator.best_params_)
            pre_comp_best.append({model_name :evolved_estimator.best_params_})
        except ValueError:
            print("hi")
    print(pre_comp_best)
    #y_test_prob = model.predict_proba(test_features)
    #y_pred = model.predict(test_features)
    #print_model_metrics(y_test_idx,y_test_prob,y_pred,label_idx)
    df = pd.DataFrame(pre_comp_best)
    df.to_csv(f"../result/all_search/{feature_name}.csv", encoding='utf-8')


# # Bag-of-Words, TF-IDF and Word Embeddings

# In[4]:


label_list= sorted(list(set(test.label.values)))
lbl_to_idx = {item:i for i,item in enumerate(label_list)}
idx_to_lbl = {i:item for i,item in enumerate(label_list)}
print(lbl_to_idx)
print(idx_to_lbl)
y_train = train.label
y_test = test.label
print(y_train.shape)
print(y_test.shape)
len(set(y_train.values))


#

# ## Bag of Words
# Let's start with simple Bag-Of-Words

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
x_train = bow.fit_transform(train.title.values)
x_test = bow.transform(test.title.values)

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="BOW")


# ## TF-IDF
#
# TFIDF should perform better than BoW since it uses document frequencies to normalize

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(train.title.values)
x_test = tfidf.transform(test.title.values)

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="TF-IDF")


# ## TF-IDF(Normalize)
#
# TFIDF should perform better than BoW since it uses document frequencies to normalize

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def things_to_unit(a):
    "if 0.5km kind of that appears, convert to unitLength etc"
    doc_units = pd.read_excel("./normalizer/units.xlsx")
    doc_dict = dict(zip(doc_units["from"],doc_units["to"]))
    for from_ in doc_dict:
        idx = np.where(
                 np.char.count(a,from_)==1
              )
        a[idx] = doc_dict[from_]
    return a

class LemmaPlaceTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`','(',')']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        val = []
        for t in word_tokenize(doc):
            if t.isdigit():
                val.append("unitN")
            elif (t not in self.ignore_tokens):
                val.append(
                    self.wnl.lemmatize(t,get_wordnet_pos(t))
                )
        new_val = np.array(val)
        new_val = np.apply_along_axis(things_to_unit, 0, new_val)
        return new_val

def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words & numbrs
    words = [word for word in words if word not in stopwords.words("english") or not word.isdigit()]


    # join words to make sentence
    document = " ".join(words)

    return document

# TF-idf w/ tokenizer
tfidf = TfidfVectorizer(tokenizer=LemmaPlaceTokenizer)
x_train = tfidf.fit_transform(train.title.values)
x_test = tfidf.transform(test.title.values)

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="TF-IDF(custom)")

# TFIDF performs marginally better than BoW. Although whats impressive here is the fact that we're getting an F1 score of 0.826 with just 50 datapoints. This is why Log Reg + TFIDF is a great baseline for NLP classification tasks.
#
# Next we'll try 100D glove vectors.

# ## GloVe

# In[7]:


# Load the glove vectors with PyMagnitude
# PyMagnitude is a fantastic library that handles a lot of word vectorization tasks.

from pymagnitude import *
from collections.abc import MutableMapping
glove = Magnitude("../vectors/glove.6B.100d.magnitude")


# In[8]:


# We'll use Average Glove here
from tqdm import tqdm_notebook
from nltk import word_tokenize


def avg_glove(df):
    vectors = []
    for title in tqdm_notebook(df.title.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)

x_train = avg_glove(train)
x_test = avg_glove(test)


# In[ ]:





#

# In[ ]:


run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="GLOVE")


# # BERT

# In[ ]:


x_train = np.genfromtxt("../datasets/train_feature_bert-base-uncased.csv",delimiter=",")
x_test = np.genfromtxt("../datasets/test_feature_bert-base-uncased.csv",delimiter=",")
y_train = np.genfromtxt("../datasets/train_label_bert-base-uncased.csv",delimiter=",")
y_test = np.genfromtxt("../datasets/test_label_bert-base-uncased.csv",delimiter=",")

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="BERT")


# # SenBERT

# In[ ]:


x_train = np.genfromtxt("../datasets/train_feature_sbert.csv",delimiter=",")
x_test = np.genfromtxt("../datasets/test_feature_sbert.csv",delimiter=",")
y_train = np.genfromtxt("../datasets/train_label_sbert.csv",delimiter=",")
y_test = np.genfromtxt("../datasets/test_label_sbert.csv",delimiter=",")

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="SENBERT")


# # Electra

# In[ ]:


x_train = np.genfromtxt("../datasets/train_feature_electra-small-discriminator.csv",delimiter=",")
x_test = np.genfromtxt("../datasets/test_feature_electra-small-discriminator.csv",delimiter=",")
y_train = np.genfromtxt("../datasets/train_label_electra-small-discriminator.csv",delimiter=",")
y_test = np.genfromtxt("../datasets/test_label_electra-small-discriminator.csv",delimiter=",")

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list,feature_name="Electra")


# # FNet

# In[ ]:


x_train = np.genfromtxt("../datasets/train_feature_fnet-base.csv",delimiter=",")
x_test = np.genfromtxt("../datasets/test_feature_fnet-base.csv",delimiter=",")
y_train = np.genfromtxt("../datasets/train_label_fnet-base.csv",delimiter=",")
y_test = np.genfromtxt("../datasets/test_label_fnet-base.csv",delimiter=",")

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list, feature_name="FNET")


# # Roberta

# In[ ]:


x_train = np.genfromtxt("../datasets/train_feature_roberta-base.csv",delimiter=",")
x_test = np.genfromtxt("../datasets/test_feature_roberta-base.csv",delimiter=",")
y_train = np.genfromtxt("../datasets/train_label_roberta-base.csv",delimiter=",")
y_test = np.genfromtxt("../datasets/test_label_roberta-base.csv",delimiter=",")

run_log_reg(x_train, x_test, y_train, y_test, lbl_to_idx=lbl_to_idx, idx_to_lbl=idx_to_lbl, label_list=label_list, feature_name="FNET")

