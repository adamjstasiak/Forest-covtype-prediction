import models_and_function as mf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 


import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

def main():

    # Task 1.

    data,columns_names = mf.read_prepare_data()
    print("Loaded data \n") 
    print(data.head())

    # Spliting data for test and train sets
    labels = data[columns_names[-1]]
    features = data[columns_names[:-1]]
    train_feature, test_feature, train_labels,test_labels = train_test_split(features,labels,test_size=0.3,random_state=1)
    test_labels = test_labels.to_numpy()
    test_labels.reshape(-1,1)

    print('Creating predictions from heuritics \n')
    heuristics_cov_type_pred = mf.heuristic(data,columns_names)    
    # decision tree
    print('Creating predictions from Decision tree \n')
    Dtree_covtype_pred = mf.Decision_tree(train_feature,train_labels,test_feature)

    # Random forest
    print('Creating predictions from Random forest \n')
    Rtree_covtype_pred = mf.Random_forest(train_feature,train_labels,test_feature)

    # Neural network

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model = mf.hp_tuner(mf.model_builder,train_feature,train_labels,stop_early)
    model_history = model.fit(train_feature,train_labels,epochs=20,validation_split=0.2,callbacks=[stop_early])

    # Plot training curves for the best hyperparameters

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('model_accuracy_history.png')

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('model_loss_history.png')

    # predict data
    nn_pred_covtype = model.predict(test_feature)
    nn_pred_covtype = np.argmax(nn_pred_covtype,axis=1)

    # Evaluating models and neural network

    # compering accuracy score

    hue_accuracy = accuracy_score(labels,heuristics_cov_type_pred)
    Dtree_accuracy = accuracy_score(test_labels,Dtree_covtype_pred)
    nn_accuracy = accuracy_score(test_labels,nn_pred_covtype)
    Rtree_accuracy = accuracy_score(test_labels,Rtree_covtype_pred)

    names = ['Hueristic','Decision_tree','Random Forest','Neural Network']
    acc  = [hue_accuracy*100,Dtree_accuracy*100,Rtree_accuracy*100,nn_accuracy*100]
    plt.figure()
    plt.bar(names,acc,width=0.4)
    plt.xlabel('Model')
    plt.ylabel('accuracy [%]')
    plt.title('Accuracy score by model')

    plt.savefig('Compering accuracy.png')

    # classification reports

    print("Classification report for Heuristics \n")
    hue_report = classification_report(labels,heuristics_cov_type_pred)
    print(hue_report,"\n")
    print("Classification report for Decision Tree \n")
    dtree_report = classification_report(test_labels,Dtree_covtype_pred)
    print(dtree_report,"\n")
    print("Classification report for Random Forest \n")
    rtree_report = classification_report(test_labels,Rtree_covtype_pred)
    print(rtree_report,"\n")
    print("Classification report for neural network \n")
    nn_report = classification_report(test_labels,nn_pred_covtype)
    print(nn_report,"\n")

    # Confusion matrix

    hue_cm = confusion_matrix(labels,heuristics_cov_type_pred)
    dtree_cm = confusion_matrix(test_labels,Dtree_covtype_pred)
    rtree_cm = confusion_matrix(test_labels,Rtree_covtype_pred)
    nn_cm = confusion_matrix(test_labels,nn_pred_covtype)


    fig,ax = plt.subplots(1,4,figsize=(20,6))
    ax[0].set_title('Heuristic')
    sns.heatmap(hue_cm,annot=True,ax=ax[0])
    ax[0].set_xlabel('Predicted labels')
    ax[0].set_ylabel('True labels')
    ax[1].set_title('Decision Tree')
    sns.heatmap(dtree_cm,annot=True,ax=ax[1])
    ax[1].set_xlabel('Predicted labels')
    ax[1].set_ylabel('True labels')
    ax[2].set_title('Random forest')
    sns.heatmap(rtree_cm,annot=True,ax=ax[2])
    ax[2].set_xlabel('Predicted labels')
    ax[2].set_ylabel('True labels')
    ax[3].set_title('Neural network')
    sns.heatmap(nn_cm,annot=True,ax=ax[3])
    ax[3].set_xlabel('Predicted labels')
    ax[3].set_ylabel('True labels')


    fig.suptitle('Confusion matrix')

    fig.savefig('Confusions_matrixes.png')
if __name__ == "__main__":
    main()