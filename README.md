## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
     import pandas as pd 
     df= pd.read_csv("/content/Encoding Data.csv")
     df
```

  <img width="477" height="464" alt="image" src="https://github.com/user-attachments/assets/b5e95247-c211-4785-8d11-32cc10e4c200" />
  
```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])
```

<img width="231" height="246" alt="image" src="https://github.com/user-attachments/assets/fbf613ba-0710-4133-b49d-a6a7b970492e" />

```
df['bo2']= e1.fit_transform(df[["ord_2"]])
df
```

<img width="484" height="468" alt="image" src="https://github.com/user-attachments/assets/c1de1a05-6f8f-487a-b3bf-a865dee84a0d" />

```
le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc
```

<img width="479" height="444" alt="image" src="https://github.com/user-attachments/assets/8d06fd3d-1b8f-460b-9a9e-ba56a96167ef" />

```

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2
 ```
<img width="643" height="454" alt="image" src="https://github.com/user-attachments/assets/9e18074c-776b-4683-b683-c2fbcbb1c203" />

```

pd.get_dummies(df2,columns=["nom_0"])


```
<img width="903" height="469" alt="image" src="https://github.com/user-attachments/assets/7eee9094-ded3-4ade-bcb0-47dfc1258b2d" />

```
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df
```
<img width="679" height="464" alt="image" src="https://github.com/user-attachments/assets/9bb71aec-4a5e-4d9a-90d2-66304febbf21" />

```
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df
```
<img width="688" height="512" alt="image" src="https://github.com/user-attachments/assets/f5574392-1b36-4af8-bd42-e175953e1583" />

```
dfb= pd.concat([df,nd],axis=1)
dfb
```
<img width="959" height="469" alt="image" src="https://github.com/user-attachments/assets/bd73dc42-750e-492a-aae8-f6c59ed144ff" />

```
from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC
```
<img width="768" height="471" alt="image" src="https://github.com/user-attachments/assets/e2a712d1-7530-49c7-8f4b-7ccfecfb4b24" />

```
import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1064" height="507" alt="image" src="https://github.com/user-attachments/assets/89a4c883-97b9-4df2-b0be-57841355e90e" />

```
df.skew()
```
<img width="437" height="246" alt="image" src="https://github.com/user-attachments/assets/007e86fe-d71f-40f5-a8fa-1b9f91b37d13" />

```
np.log(df["Highly Positive Skew"])
```
<img width="407" height="564" alt="image" src="https://github.com/user-attachments/assets/48faeaeb-038b-4d2d-87ff-9f1f11e3ee57" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="408" height="511" alt="image" src="https://github.com/user-attachments/assets/272afabc-ae1d-455b-b4ff-9471a5c41812" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="361" height="518" alt="image" src="https://github.com/user-attachments/assets/3f14749a-1f4c-4a5d-9407-3fb58fca83fe" />

```
np.square(df["Highly Positive Skew"])
```
<img width="387" height="520" alt="image" src="https://github.com/user-attachments/assets/2da30cd2-37a9-470b-8c92-b47dbbbe4a59" />

```
df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1382" height="524" alt="image" src="https://github.com/user-attachments/assets/e4650163-6664-4283-877a-fc02f76f8fb6" />

```
df.skew()
```
<img width="484" height="301" alt="image" src="https://github.com/user-attachments/assets/20ce3e67-858c-47db-8ec8-86d82e82f3d8" />

```
df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="536" height="353" alt="image" src="https://github.com/user-attachments/assets/483548c4-ae7b-4f32-9f9e-2213d2d4ad91" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])
 df
```
<img width="1453" height="586" alt="image" src="https://github.com/user-attachments/assets/67422ebb-5314-4099-a381-df071952d0a3" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
<img width="773" height="556" alt="image" src="https://github.com/user-attachments/assets/dfb1e6b8-73b0-4f92-ad18-c6115fd179ae" />

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
<img width="747" height="524" alt="image" src="https://github.com/user-attachments/assets/e28e182f-6cef-4e3e-855f-f898e541c9a7" />

```

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```
<img width="708" height="541" alt="image" src="https://github.com/user-attachments/assets/b6c58716-bf4d-4105-bea2-36ec4bbc68d8" />

```

 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()

```
<img width="739" height="550" alt="image" src="https://github.com/user-attachments/assets/67ef64c8-cc9c-45df-a613-b98db7aa6e55" />

```
dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

```
<img width="708" height="523" alt="image" src="https://github.com/user-attachments/assets/4ec9de4e-10f4-4131-a759-d89afd8179c1" />

```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```
<img width="706" height="527" alt="image" src="https://github.com/user-attachments/assets/eb0c56de-d408-4137-911f-f9a6ba9b8fc6" />

# RESULT:
           Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully


       
