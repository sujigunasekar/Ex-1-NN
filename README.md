<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```
## OUTPUT:![Screenshot 2024-08-23 081506](https://github.com/user-attachments/assets/d3eaec06-de65-4266-b4aa-20d20f3ccd9b)
![Screenshot 2024-08-23 081609](https://github.com/user-attachments/assets/56b0b99f-a144-4666-ac88-e582d2c5703e)
![Screenshot 2024-08-23 081644](https://github.com/user-attachments/assets/1f928434-292c-4f73-a788-218c9841b177)
![Screenshot 2024-08-23 091804](https://github.com/user-attachments/assets/71c51224-7f1f-4343-aea2-4447adb7c768)
![Screenshot 2024-08-23 091843](https://github.com/user-attachments/assets/459f1069-bd95-45e0-9503-c286d124cd9f)
![Screenshot 2024-08-23 091948](https://github.com/user-attachments/assets/68f444bc-1183-43b1-bc0b-3709fd0d5f58)
![Screenshot 2024-08-23 092033](https://github.com/user-attachments/assets/74055cc2-4f8a-4420-8597-549f129aaa72)
![Screenshot 2024-08-23 092116](https://github.com/user-attachments/assets/e9c25fa5-c19d-465b-ad90-54d530f275ea)
![Screenshot 2024-08-23 092156](https://github.com/user-attachments/assets/16370c25-5457-403c-ba02-436e3ec879f3)
![Screenshot 2024-08-23 092221](https://github.com/user-attachments/assets/81534f56-3b47-4062-bda1-079a5ea26ab8)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


