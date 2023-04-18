import numpy as np
import pandas as pd
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import keras_tuner as kt

def read_prepare_data():
    """
    Read Covertype Data Set from covtype.data and add column labels
    to pandas DataFrame. Column names based on documentation in covtype.info
    Return data frame which contains covtype data and column labels.
    """
    columns_names = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
    wild = 'Wilderness Areas type '
    for i in range(1,5):
        columns_names.append((wild+str(i)))
    soils = 'Soil Type '
    for i in range(1,41):
        columns_names.append((soils+str(i)))
    forest = 'Forest Cover Type'
    columns_names.append(forest)    
    data = pd.read_csv('covtype.data',sep=',',names=columns_names)
    return data,columns_names

# task 2 simple heuritics

def heuristic(data,columns_names):
    """
    Simple heuristic to classify data based on Relevant Information Paragraph in covtype.info 
    Will Classify cover type based on their Wilderness Areas
    Parameters:
        Input data and column labels

    Return predicted values
    """
    class_data = data[columns_names[14:]]
    Cache = class_data[columns_names[-1]].to_list()
    Comanche = class_data[columns_names[-2]].to_list()
    Neota = class_data[columns_names[-3]].to_list()
    Rewah = class_data[columns_names[-4]].to_list()
    cov_type_pred = np.array([7 for i in range(len(Cache))])
    cov25 = [2,5]
    cov346 = [3,4,6]   
    for i in range(len(Cache)):
        if Neota[i] == 1 and Rewah[i]==0 and Comanche[i] ==0 and Cache == 0:
            cov_type_pred[i] = 1
        elif (Neota[i] == 0 and Rewah[i]==1 and Comanche[i] == 0 and Cache == 0) or ( Neota[i] == 0 and Rewah[i]==0 and Comanche[i] ==1 and Cache == 0):
            cov_type_pred[i] =random.choice(cov25)
        elif Neota[i] == 0 and Rewah[i]==0 and Comanche[i] == 0 and Cache ==1:
            cov_type_pred[i]  = random.choice(cov346)
    return np.array(cov_type_pred)

# task 3 Machine learnig models from sklearn library

def Decision_tree(train_feature,train_labels,test_feature):
    """
    Decision tree model for covtype dataset
    Parameters:
        train_feature - features used to train model
        train_labels - labels used to train model
        test_feature - features used to make predict on our model
        test_labels - labels used to check our model
    
    Return:
       Dtree_covtype_pred predicted values
    """
    Dtree = DecisionTreeClassifier()
    Dtree.fit(train_feature,train_labels)
    Dtree_covtype_pred = Dtree.predict(test_feature)
    return Dtree_covtype_pred
    

def Random_forest(train_feature,train_labels,test_feature):
    """
    Random forest model for covtype dataset
    Parameters:
        train_feature - features used to train model
        train_labels - labels used to train model
        test_feature - features used to make predict on our model
        test_labels - labels used to check our model
    
    Return:
        Rtree_covtype_pred - predicted values
    """
    
    Rtree = RandomForestClassifier()
    Rtree.fit(train_feature,train_labels)
    Rtree_covtype_pred = Rtree.predict(test_feature)
    return Rtree_covtype_pred

def model_builder(hp):
    """
    Function creating sequentail tensorflow model for neural network
    Parameters:
        hp- hyperparametres we obtain from optimazer
    Return 
        model with hyperparameters
    """

    model = tf.keras.Sequential()
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=100, step=1)
    hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=100, step=1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation,input_dim = 54))
    model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))

    model.add(tf.keras.layers.Dense(units=20, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
  
    return model

def hp_tuner(model_builder,x_train,y_train,stop_early):
    """
    Hyper parameters tuner basic on model builder. 
    Parameter:
        model_builder - function crating a model
        x_train - features train set
        y_train - lable train set 
        stop_early - callback to terminated tuner when we don't have improvment
    Return:
        Model mwith proper parameters
    """
    tuner = kt.Hyperband(model_builder,objective='val_accuracy',max_epochs=10,factor=3,directory='dir',project_name='x')
    tuner.search(x_train,y_train,epochs=20,validation_split=0.2,callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    return model