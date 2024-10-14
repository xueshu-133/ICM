# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:33:20 2022

@author: qiubinxu
"""

import streamlit as st
import pandas as pd #处理数据所用库
import numpy as np
import requests #工具，访问服务器
import numpy as np#加载数据所用库
import pandas as pd #处理数据所用库
import xgboost
import xgboost as xgb#极限梯度提升机所用库
from xgboost import XGBClassifier#分类算法#加载极限梯度提升机中分类算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree #导入需要的模块
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#模型选择将数据集分为测试集和训练集
from sklearn.metrics import accuracy_score#模型最终的预测准确度分数
from sklearn.datasets import load_iris#加载鸢尾花集数据
from sklearn.datasets import load_boston#加载鸢尾花集数据
from sklearn.datasets import load_breast_cancer#加载鸢尾花集数据
import matplotlib#加载绘图工具
from xgboost import plot_importance#加载极限梯度提升机中重要性排序函数
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score   # 准确率
import scipy.stats as stats
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as CVS
from xgboost import plot_importance
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from IPython.display import display, Image
from sklearn.model_selection import cross_val_score,StratifiedKFold
st.title('A Machine learning based online calculator for predicting the risk of ICM'.center(33, '-'))
classes = {0:'NLM',1:'LM'}
st.sidebar.expander('')
st.sidebar.subheader('Variable') 
Sex=st.sidebar.selectbox('Sex', ['Male','Female'])
Sex_map = {'Male':0,'Female':2}
Hypertension=st.sidebar.selectbox('Hypertension', ['Yes','No'])
Hypertension_map = {'Yes':1,'No':0}
Diabetes=st.sidebar.selectbox('Diabetes', ['Yes','No'])
Diabetes_map = {'Yes':1,'No':0}
Age = st.sidebar.slider('Age:',
                          min_value=0,
                          max_value=100)
Hemoglobinm = st.sidebar.number_input("Hemoglobinm")
BMI = st.sidebar.number_input("BMI")
TyG_BMI= st.sidebar.number_input("TyG-BMI")
TC = st.sidebar.number_input("TG")
HDL_C = st.sidebar.number_input("HDL-C")
TC_HDL = st.sidebar.number_input("TG/HDL-C")
Ejection_Fraction = st.sidebar.number_input("Ejection-Fraction")
filename = 'modelDZ.txt' 
x = []
x.extend([Sex_map[Sex],Hypertension_map[Hypertension],Diabetes_map[Diabetes],
         Age,Hemoglobinm,BMI,TyG_BMI,TC,HDL_C,TC_HDL,Ejection_Fraction])
x=np.array(x).reshape(1,11)
import pickle
if st.button("Predict"):
    #  predict_class()
    import os
    if os.path.exists(filename):
        with open(filename, 'rb') as fq:
            modelXGB = pickle.load(fq, encoding='bytes')
            y_pred = modelXGB.predict_proba(x)
            print(max(y_pred[:,1]))
            st.header('The precentage of ICM progression is: %.2f %%' % (max(y_pred[:,1])* 100))



