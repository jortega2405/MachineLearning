#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:07:52 2022

@author: joseortega
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('./DataSets/weatherAUS.csv')

data = data.set_index('Date')
data.Location.replace(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle',
                       'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond','Sydney', 'SydneyAirport',
                       'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini',
                       'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil',
                       'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville',
                       'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe',
                       'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 
                       'Launceston','AliceSprings', 'Darwin', 'Katherine', 'Uluru'], [0, 1, 2, 3, 4, 5, 6, 7, 8
                                                                                  , 9, 10, 11, 12, 13, 14, 15, 16
                                                                                  , 17, 18, 19, 20, 21, 22, 23, 24
                                                                                  , 25, 26, 27, 28, 29, 30, 31, 32
                                                                                  , 33, 34, 35, 36, 37, 38, 39, 40
                                                                                  , 41, 42, 43, 44, 45, 46, 47, 48],
inplace = True)
data.MaxTemp.replace(np.nan, 23.23, inplace=True)
data.MinTemp.replace(np.nan, 12.19, inplace=True)
data.Evaporation.replace(np.nan, 5.47, inplace = True)
data.Rainfall.replace(np.nan, 2.35, inplace = True)
data.Sunshine.replace(np.nan, 7.62, inplace = True)
data.WindGustDir.replace(np.nan, -1, inplace = True)
data.WindDir9am.replace(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'nan', 'SSW',
                         'N', 'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                        inplace = True)
data.WindGustDir.replace(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
       'S', 'NW', 'SE', 'ESE', 'nan' , 'E', 'SSW'], [0, 1, 2, 3, 4, 5, 6, 7, 8,9 ,10
                                                 , 11, 12, 13, 14, 15, 16], inplace = True)
data.WindDir3pm.replace(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
                         'SW', 'SE', 'N', 'S', 'NNE', 'nan', 'NE'],
                        [1, 11, 13, 14, 0, 5, 12, 3, 1, 9, 4, 2, 10, 6, 16, 8, 7]
                        , inplace = True)
data.WindSpeed9am.replace(np.nan, 14.0, inplace = True)
data.WindSpeed3pm.replace(np.nan, 18.64, inplace = True)
data.Humidity9am.replace(np.nan, 68.84, inplace = True)
data.Humidity3pm.replace(np.nan, 51.48, inplace = True)
data.Pressure9am.replace(np.nan, 1017.65, inplace = True)
data.Pressure3pm.replace(np.nan, 1015.26, inplace = True)
data.Cloud9am.replace(np.nan, 4.44, inplace = True)
data.Cloud3pm.replace(np.nan, 4.50, inplace = True)
data.Temp9am.replace(np.nan, 16.99, inplace = True)
data.Temp3pm.replace(np.nan, 21.69, inplace = True)
data.RISK_MM.replace(np.nan, 2.36, inplace = True)
data.RainToday.replace(['No', 'Yes'], [0, 1], inplace = True)
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace = True)
data.dropna(axis=0, how='any', inplace=True)


def entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


def metrics(str_model, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))

x = np.array(data.drop(['RainTomorrow'], 1))
y = np.array(data.RainTomorrow)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

model1 = DecisionTreeClassifier()
model1, accuracy_test1, accuracy1, y_predict1 = entrenamiento(model1, x_train, x_test, y_train, y_test)
metrics('Arbol de decisión', accuracy_test1, accuracy1, y_test, y_predict1)

model2 = KNeighborsClassifier()
model2, accuracy_test2, accuracy2, y_predict2 = entrenamiento(model2, x_train, x_test, y_train, y_test)
metrics('Vecinos mas cercanos', accuracy_test2, accuracy2, y_test, y_predict2)

model3 = LogisticRegression(solver='lbfgs', max_iter=5000)
model3, accuracy_test3, accuracy3, y_predict3 = entrenamiento(model3, x_train, x_test, y_train, y_test)
metrics('Regresion Logistica', accuracy_test3, accuracy3, y_test, y_predict3)

model4 = GaussianNB()
model4, accuracy_test4, accuracy4, y_predict4 = entrenamiento(model4, x_train, x_test, y_train, y_test)
metrics('Gaussian NB', accuracy_test4, accuracy4, y_test, y_predict4)

model5 = RandomForestClassifier()
model5, accuracy_test5, accuracy5, y_predict5 = entrenamiento(model5, x_train, x_test, y_train, y_test)
metrics('Random Forest Clasifier', accuracy_test5, accuracy5, y_test, y_predict5)











































