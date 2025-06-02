# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:

Read the given Data.

STEP 2:

Clean the Data Set using Data Cleaning Process.

STEP 3:

Apply Feature Encoding for the feature in the data set.

STEP 4:

Apply Feature Transformation for the feature in the data set.

STEP 5:

Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : VISHAL S
### Reg No : 212222040181

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/6a654a7f-8923-43af-837f-fbaaae172990)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/da24dfe7-2a7e-4065-9fcf-a4eca0bc2bd2)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/101cabae-f64c-4b1d-bc3d-1ced6bea5092)


```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/0e5b50a7-166d-4c8a-aa7f-47d4a829154d)

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/737d95d4-7f40-4c73-a428-e710982d509e)


```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/9e7e2796-d572-42bf-9a59-799e6105bb68)


```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/4f6f6a5a-3ed7-42b2-893d-ef09c752be3d)


```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/f70c4873-aca1-4a62-a7a3-b40dfe168af1)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/35a6bcdc-2928-41c5-986c-903d5205d9d5)


```py
df.skew()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/9cebcdc2-ff61-4e31-bc34-db6b3388e546)


```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/c316b1f8-a5a2-4487-99a9-1a0144e00ae1)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/d5ccd5c9-87cb-42e4-8420-a37ea75e9add)


```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/09d3e8f9-0558-4039-90ff-3b358dcaeb2c)


```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/a1ab31eb-09c6-4bb6-a569-466254420fee)


```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/4133ceae-19e0-487a-9e04-2185421e999c)


```py
df.skew()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/f2b6800d-4fda-4da3-9652-f3dcc47dc76d)


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/84134adb-72d1-4afc-bf2f-6649e0a00a6e)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/d8838866-2c73-4e6f-bb78-719723589804)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/48cf7384-276f-4955-8bad-16ef34e4b075)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/9a9a16d7-5d01-4216-bee8-255e7477463f)



```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/d4a59337-cc27-46c6-b47c-5e9ba7988626)


```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/6f7a4eaa-1c54-4409-8b57-8b407d5842f7)


```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/e2ff6572-cb52-434f-8d9a-980843e1a1b9)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/d5c66705-7e21-4a6b-8bc7-a9e1ca23ae85)



## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
