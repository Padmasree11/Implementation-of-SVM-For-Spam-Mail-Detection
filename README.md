# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/
import pandas as pd

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("spam EX 12 ML.csv", encoding='Windows-1252')

print(data.head())

print("Dataset Shape:", data.shape)

x = data['v2'].values
y = data['v1'].values

print("x shape:", x.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("Predictions:")
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(acc)

con = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(con)

cl = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(cl)
```

## Output:
<img width="1003" height="673" alt="image" src="https://github.com/user-attachments/assets/889011e2-ed67-48a1-b473-e975827c03ba" />
<img width="653" height="562" alt="image" src="https://github.com/user-attachments/assets/55ce8f27-80dc-4be8-9ec2-5b4538e6db2a" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
