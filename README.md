# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rithik M
RegisterNumber: 212225040342 
*/
```

```
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

data = pd.read_csv('spam.csv', encoding='Windows-1252')
x = data["v2"].values
y = data["v1"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
```

## Output:
<img width="885" height="185" alt="Screenshot 2026-03-19 104231" src="https://github.com/user-attachments/assets/7a850d8c-0f7f-4e8b-8ad9-309a1e155cc8" />
<img width="909" height="296" alt="Screenshot 2026-03-19 104239" src="https://github.com/user-attachments/assets/ed108406-66aa-411e-ba86-669aa1f5404a" />
<img width="519" height="342" alt="Screenshot 2026-03-19 104249" src="https://github.com/user-attachments/assets/f2c7fb71-9fff-4226-9f65-f80eec198606" />
<img width="275" height="212" alt="Screenshot 2026-03-19 104300" src="https://github.com/user-attachments/assets/76c3492f-2ca1-45ea-93df-774d25350ce0" />
<img width="814" height="227" alt="Screenshot 2026-03-19 104309" src="https://github.com/user-attachments/assets/008e8d2b-2a62-4a56-9e5f-d9e16382902c" />
<img width="525" height="164" alt="Screenshot 2026-03-19 104603" src="https://github.com/user-attachments/assets/64b8264a-97e1-497c-9464-f179a2895706" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
