import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from keras import models, regularizers, layers, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
print("Modules imported \n")

#Print Confusion MAtrix for data
def show_confusion_matrix(confusion_mx, targets, title='Confusion Matrix', normalize=False):
  accuracy = np.trace(confusion_mx) / float(np.sum(confusion_mx))
  misclass = 1 - accuracy
  cmap = plt.get_cmap('Oranges')
  plt.figure(figsize=(8, 6))
  plt.imshow(confusion_mx, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  ticks = np.arange(len(targets))
  plt.xticks(ticks, targets, rotation=45)
  plt.yticks(ticks, targets)  
  threshold = confusion_mx.max() / 2
  for i, j in itertools.product(range(confusion_mx.shape[0]), range(confusion_mx.shape[1])):
    plt.text(j, i, "{:,}".format(confusion_mx[i, j]),horizontalalignment="center",color="white" if confusion_mx[i, j] > threshold else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

# Get ROC and AUC Score for model
def final_roc_auc_score(y_test, end_predictions, average="macro"):
  lb = LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_predictions = lb.transform(end_predictions)
  return roc_auc_score(y_test, y_predictions, average=average)

def main():
  raw_data = pd.read_csv('raw_data.csv') #read data into dataframe
  data_no_id = raw_data.drop('hadm_id', 1) #drop id column from frame

  data_no_id.info() #show data info
  data_no_id.describe()
  print(data_no_id.head(15))

  y = data_no_id['LOSgroupNum'] #seperate targets. LOS Grouped by 0.0-4.0, 4.0-8.0, 8.0-12.0, 12.0+ 
  X = data_no_id.drop(['LOSgroupNum'], 1)
  X = X.drop(['LOSdays', 'ExpiredHospital', 'AdmitDiagnosis', 'AdmitProcedure', 'marital_status', 'ethnicity', 'religion', 'insurance'], 1) #drop columns that are hard to convert to numbers, or irrelevant

  categorical_columns = ['gender', 'admit_type', 'admit_location'] #columns to convert

  X_pca = X.drop(categorical_columns, axis=1)
  pca = PCA(n_components=14)
  X_pca = pd.DataFrame(data=pca.fit_transform(X_pca))

  for col in categorical_columns: #for each of the columns drop it and make one hot vectors
  
    if col in X.columns:
      temp_one_hot = pd.get_dummies(X[col])
      X = X.drop(col, axis=1)
      X = X.join(temp_one_hot, lsuffix='_left', rsuffix='_right')
      X_pca = X_pca.join(temp_one_hot, lsuffix='_left', rsuffix='_right')

  X = X_pca
  X_not_standardized = X.copy()

  temp_X = X_not_standardized.values
  scaler = preprocessing.StandardScaler()
  X_scaled = scaler.fit_transform(temp_X) #Standardize data
  XNorm = pd.DataFrame(X_scaled, columns=X_not_standardized.columns)

  X_train, X_test, y_train, y_test = train_test_split(XNorm, y, test_size=0.2, random_state=42) #split into training and and testing sets

  input_y_train = to_categorical(y_train) #convert to binary classes
  y_val = to_categorical(y_test)

  model = models.Sequential() #Create NN Model
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(29,)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(4, activation='softmax'))
  print(model.summary())

  EPOCHS = 70
  BATCH_SIZE = 16
  model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  history = model.fit(X_train, input_y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_val))

  results = model.evaluate(X_test, y_val)
  print("___" * 25)
  print("Test Loss and Accuracy")
  print("Results ", results)
  history_dictionary = history.history
  history_dictionary.keys()

  plt.clf() #plot training and validation loss
  loss_values = history_dictionary['loss']
  validation_loss_values = history_dictionary['val_loss']
  epochs = range(1, (len(history_dictionary['loss']) + 1))
  plt.plot(epochs, loss_values, 'bo', label='Training loss')
  plt.plot(epochs, validation_loss_values, 'r', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs: ')
  plt.ylabel('Loss: ')
  plt.legend()
  plt.show()

  plt.clf() #plot training and validation accuarcy
  accuracy_values = history_dictionary['categorical_accuracy']
  validation_accuracy_values = history_dictionary['val_categorical_accuracy']
  epochs = range(1, (len(history_dictionary['categorical_accuracy']) + 1))
  plt.plot(epochs, accuracy_values, 'bo', label='Training accuracy')
  plt.plot(epochs, validation_accuracy_values, 'r', label='Validation accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs: ')
  plt.ylabel('Accuracy: ')
  plt.legend()
  plt.show()

  end_predictions = model.predict(X_test) #test model

  # print(end_predictions)
  temp_predictions = []
  count = end_predictions.shape[0]
  for i in range(count): #for each prediction finds the class from the output... uncomment print statements if confused
    temp_predictions.append(np.argmax(end_predictions[i])) 
  predictions = np.array(temp_predictions)  
  # print(predictions)

  temp_predictions = []
  count = y_val.shape[0]
  for i in range(count): #for each prediction finds the class from the output 
    temp_predictions.append(np.argmax(y_val[i])) 
  y_val_numbers = np.array(temp_predictions)  

  confusion_mx = confusion_matrix(y_val_numbers, predictions) #get confusion matrix
  show_confusion_matrix(confusion_mx, targets = [0,1,2,3]) #create matrix

  #Figure out more stats
  NUMBER_OF_CLASSES = 4

  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0

  for los_class in range(NUMBER_OF_CLASSES):
    sum_confusion_mx = np.sum(confusion_mx)
    temp_true_positives = confusion_mx[los_class,los_class]
    temp_false_negatives = np.sum(confusion_mx[los_class,:], axis=0) - temp_true_positives
    temp_false_positives = np.sum(confusion_mx[:,los_class], axis=0) - temp_true_positives
    temp_true_negatives = sum_confusion_mx - (temp_true_positives + temp_false_negatives + temp_false_positives)
    print('Class ',los_class)

    temp_confusion_mx = np.zeros([2, 2], dtype=np.int32)
    temp_confusion_mx[0,0] = temp_true_negatives
    temp_confusion_mx[0,1] = temp_false_positives
    temp_confusion_mx[1,0] = temp_false_negatives
    temp_confusion_mx[1,1] = temp_true_positives #bottom right because 1 is true and 0 is false

    show_confusion_matrix(temp_confusion_mx,targets = [0,1],title = "Confusion Matrix For Class " + str(los_class)) #show graph

    accuracy = (temp_true_positives + temp_true_negatives) / (temp_true_positives + temp_true_negatives + temp_false_positives + temp_false_negatives)
    recall = temp_true_positives / (temp_true_positives + temp_false_negatives)
    precision = temp_true_positives / (temp_true_positives + temp_false_positives)
    f1_score = 2 * recall * precision / (recall + precision)
    
    #print stats
    print('True Positives ',temp_true_positives)
    print('False Negatives ',temp_false_negatives)
    print('False Positives ',temp_false_positives)
    print('True Negatives ',temp_true_negatives)
    print('sum ', temp_true_positives + temp_true_negatives + temp_false_positives + temp_false_negatives)
    print(temp_confusion_mx)
    print('Sum of CM ', np.sum(temp_confusion_mx))
    print ('Accuracy ',round(accuracy, 4))
    print('Recall ', round(recall, 4))
    print('Precision ', round(precision, 4))
    print('F1 Score ', round(f1_score, 4))
    print('---' * 25)
    
    true_positives += temp_true_positives
    true_negatives += temp_true_negatives
    false_positives += temp_false_positives
    false_negatives += temp_false_negatives

  print('All Classes')

  print('AUC ROC Score: ', final_roc_auc_score(y_val_numbers, predictions))
  print('---' * 25)

main()