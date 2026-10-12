#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


import warnings 
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# In[25]:


df = pd.read_csv("AIML Dataset.csv")


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


df.columns


# In[29]:


df["isFraud"].value_counts()


# In[30]:


df["isFlaggedFraud"].value_counts()


# In[31]:


df.isnull().sum().sum()


# In[32]:


df.shape[0]


# In[33]:


round((df["isFraud"].value_counts()[1]/df.shape[0]) * 100,2)


# In[34]:


df["type"].value_counts().plot(kind="bar", title="Transaction Types", color = "skyblue")
plt.xlabel("Transaction Types")
plt.ylabel("Count")
plt.show()


# In[22]:


fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
fraud_by_type.plot(kind="bar", title="Fraud Rate by Type", color="salmon")
plt.ylabel("Fraud Rate")
plt.show()


# In[36]:


df["amount"].describe().astype(int)


# In[38]:


sns.histplot(np.log1p(df["amount"]), bins=100, kde=True, color="green")
plt.title("Transaction Amount Distribution (log scale)")
plt.xlabel("Log(Amount + 1)")
plt.show()


# In[39]:


sns.boxplot(data=df[df["amount"]<50000], x="isFraud", y="amount")
plt.title("Amount vs isFraud (Filtered under 50k)")
plt.show()


# In[40]:


df.columns


# In[41]:


df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]


# In[43]:


(df["balanceDiffOrig"] < 0).sum()


# In[44]:


(df["balanceDiffDest"] < 0).sum()


# In[47]:


df.head(2)


# In[ ]:


frauds_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
plt.plot(frauds_per_step.index, frauds_per_step.values, label="Frauds Per Step")
plt.xlabel("Step (Time)")
plt.ylabel("Number of Frauds")
plt.title("Frauds Over Time")
plt.grid(True)
plt.show()


# In[49]:


df.drop(columns = "step", inplace=True)


# In[50]:


df.head()


# In[51]:


top_senders = df["nameOrig"].value_counts().head(10)


# In[52]:


top_senders


# In[53]:


top_receivers = df["nameDest"].value_counts().head(10)


# In[55]:


top_receivers


# In[56]:


fraud_users = df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)


# In[57]:


fraud_users


# In[ ]:


fraud_types = df[df["type"].isin(["TRANSFER","CASH_OUT"])]


# In[60]:


fraud_types["type"].value_counts()


# In[62]:


sns.countplot(data=fraud_types, x="type", hue="isFraud")
plt.title("Fraud Detection in Transfer & Cash_Out")
plt.show()


# In[63]:


corr = df[["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud"]].corr()


# In[64]:


corr


# In[65]:


sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[66]:


zero_after_transfer = df[
    (df["oldbalanceOrg"]>0) &
    (df["newbalanceOrig"]==0) &
    (df["type"].isin(["TRANSFER","CASH_OUT"]))
]


# In[67]:


len(zero_after_transfer)


# In[68]:


zero_after_transfer.head()


# In[69]:


df["isFraud"].value_counts()


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[71]:


df.head()


# In[72]:


df_model = df.drop(["nameOrig","nameDest","isFlaggedFraud"], axis = 1)


# In[73]:


df_model.head()


# In[74]:


categorical = ["type"]
numeric = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]


# In[75]:


y = df_model["isFraud"]
x = df_model.drop("isFraud", axis = 1)


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.3, stratify=y, random_state=42)


# In[77]:


preprocessor = ColumnTransformer(
    transformers= [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)


# In[79]:


pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])


# In[80]:


pipeline.fit(x_train, y_train)


# In[82]:


y_pred = pipeline.predict(x_test)


# In[84]:


print(classification_report(y_test, y_pred))


# In[85]:


confusion_matrix(y_test, y_pred)


# In[87]:


pipeline.score(x_test, y_test) * 100


# In[88]:


import joblib

joblib.dump(pipeline,"fraud_detection_pipeline.pkl")


# In[ ]:




