#!/usr/bin/env python
# coding: utf-8

# # Crime Arrest Data Analysis

# Domain: Public Safety / Crime Analytics
# 
# Objective: Analyze arrest records to identify demographic trends, crime patterns, geographic hotspots, and temporal trends using Python-based exploratory data analysis and visualization techniques.

# 1.Law enforcement agencies collect large volumes of arrest data.
# 
# 2.Raw records contain missing values, inconsistent formats, and redundant information.
# 
# 3.Goal is to preprocess the data and uncover patterns related to age, gender, area, time, and charge categories.
# 
# Dataset details:
# 
# Source: Los Angeles Open Data
# 
# Records: ~70,000
# 
# Columns: 24
# 
# Mixed numerical, categorical, and temporal features.
# 
# This project uses a real-world arrest dataset from Los Angeles containing approximately 70,000 arrest records with 24 features, including demographic information, arrest details, geographic coordinates, and booking information.
# 
# 

# # Importing the libraries

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

print('Libraries imported successfully')


# # Load dataset
# 

# In[3]:


df = pd.read_csv('C:/Users/malav/Downloads/Crime data.csv')

# Display first 5 rows
df.head()


# Number of rows and columns

# In[4]:


print('Rows:', df.shape[0])
print('Columns:', df.shape[1])


# Column names

# In[5]:


df.columns


# Data types

# In[6]:


df.dtypes


# Info

# In[7]:


df.info()


# Statistical summary

# In[8]:


df.describe(include='all')


# # Initial observations
# 
# 

# 1.Date columns are stored as text.
# 
# 2.Several categorical columns contain missing values.
# 
# 3.Geographic coordinates are available.
# 
# 4.Time is stored as HHMM integer format.

# # Data Cleaning & Pre-processing

# 1.Remove Duplicates

# In[9]:


print('Duplicate rows:', df.duplicated().sum())

df = df.drop_duplicates()

print('After removing duplicates:', df.shape)


# 2.Convert Date Columns

# In[10]:


df['Arrest Date'] = pd.to_datetime(df['Arrest Date'], errors='coerce')
df['Booking Date'] = pd.to_datetime(df['Booking Date'], errors='coerce')

df[['Arrest Date', 'Booking Date']].head()


# 3.Handle Missing Values

# In[11]:


missing = df.isnull().sum().sort_values(ascending=False)
missing[missing > 0]


# In[12]:


missing_percent = (df.isnull().sum() / len(df)) * 100
missing_percent.sort_values(ascending=False)


# Fill important categorical columns.

# In[13]:


cat_cols = ['Sex Code', 'Descent Code', 'Charge Group Description',
            'Charge Description', 'Disposition Description',
            'Booking Location']

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')


# Fill numerical values.

# In[14]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# 4.Rename Columns

# In[15]:


df.columns = df.columns.str.replace(' ', '_')

df.columns


# 5.Remove Unnecessary Columns

# In[16]:


df = df.drop(columns=['Location'], errors='ignore')


# NB:The Location column was removed because latitude and longitude provide the same information in a more usable numerical format.

# 6.Create Derived Features

# Arrest Year

# In[17]:


df['Arrest_Year'] = df['Arrest_Date'].dt.year


# Arrest Month

# In[18]:


df['Arrest_Month'] = df['Arrest_Date'].dt.month_name()


# Arrest Day

# In[19]:


df['Arrest_Day'] = df['Arrest_Date'].dt.day_name()


# Arrest Hour

# In[20]:


df['Time'].head(20)


# In[21]:


df['Time'].unique()[:20]


# Check Data type

# In[22]:


df['Time'].dtype


# Find the problematic values

# In[23]:


df[~df['Time'].astype(str).str.match(r'^\d+$', na=False)]['Time'].unique()


# In[24]:


# Convert Time to numeric, invalid values become NaN
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

# Fill missing values with 0
df['Time'] = df['Time'].fillna(0).astype(int)

# Convert to 4-digit string
df['Time'] = df['Time'].astype(str).str.zfill(4)

# Extract hour
df['Arrest_Hour'] = df['Time'].str[:2].astype(int)

df[['Time', 'Arrest_Hour']].head()


# In[25]:


df[['Time', 'Arrest_Hour']].sample(10)


# # Exploratory Data Analysis (EDA)

# In[26]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# In[27]:


df.nunique().sort_values(ascending=False)


# In[28]:


print("Report Types:", df['Report_Type'].unique())
print("Sex Codes:", df['Sex_Code'].unique())
print("Area Names:", df['Area_Name'].nunique())
print("Charge Groups:", df['Charge_Group_Description'].nunique())


# Summary Statistics

# In[29]:


df.describe()


# In[30]:


df.describe(include='object')


# Top 10 Arrest Areas

# In[31]:


df['Area_Name'].value_counts().head(10)


# Top Charge Categories

# In[32]:


df['Charge_Group_Description'].value_counts().head(10)


# Arrests by Gender

# In[33]:


df['Sex_Code'].value_counts()


# Arrest Type

# In[34]:


df['Arrest_Type_Code'].value_counts()


# # GroupBy Analysis

# Arrests by Area

# In[35]:


area_counts = df.groupby('Area_Name').size().sort_values(ascending=False)

area_counts.head(10)


# Average Age by Gender

# In[38]:


df.groupby('Sex_Code')['Age'].mean().round().astype(int)


# Average Age by Area

# In[39]:


df.groupby('Area_Name')['Age'].mean().sort_values(ascending=False).astype(int)


# Arrests by Month

# In[41]:


month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

df['Arrest_Month'] = pd.Categorical(
    df['Arrest_Month'],
    categories=month_order,
    ordered=True
)

df.groupby('Arrest_Month').size()


# Arrests by Hour

# In[42]:


df.groupby('Arrest_Hour').size()


# # Pivot Table Analysis

# Average Age by Gender and Arrest Type

# In[46]:


pivot_age = pd.pivot_table(
    df,
    values='Age',
    index='Sex_Code',
    columns='Arrest_Type_Code',
    aggfunc='mean'
)

pivot_age = pivot_age.round().astype('Int64')

pivot_age


# Arrest Count by Area and Gender

# In[47]:


pivot_area = pd.pivot_table(
    df,
    values='Report_ID',
    index='Area_Name',
    columns='Sex_Code',
    aggfunc='count',
    fill_value=0
)

pivot_area.head()


# # Crosstab Analysis

# Gender vs Arrest Type

# In[48]:


pd.crosstab(df['Sex_Code'], df['Arrest_Type_Code'])


# Area vs Report Type

# In[49]:


pd.crosstab(df['Area_Name'], df['Report_Type'])


# # Correlation Analysis

# Select only numerical columns.

# In[50]:


num_df = df.select_dtypes(include=['int64', 'float64'])

corr = num_df.corr()

corr


# This will later be used to create the heatmap.

# Oldest Person Arrested

# In[51]:


df.nlargest(5, 'Age')[['Age', 'Area_Name', 'Charge_Group_Description']]


# Youngest Person Arrested

# In[53]:


df.nsmallest(5, 'Age')[['Age', 'Area_Name', 'Charge_Group_Description']]


# In[54]:


(df['Age'] == 0).sum()


# In[55]:


zero_age = (df['Age'] == 0).sum()

print("Age = 0 records:", zero_age)
print("Percentage:", round((zero_age / len(df)) * 100, 2), "%")


# In[56]:


df = df[df['Age'] > 0]


# In[57]:


print("Age = 0 records:", (df['Age'] == 0).sum())
print("New dataset shape:", df.shape)


# NB:Records with Age = 0 were removed because they likely represent missing or invalid age values rather than actual ages.

# In[60]:


df.nsmallest(10, 'Age')[['Age', 'Area_Name', 'Charge_Group_Description']]


# Areas with Maximum Arrests

# In[61]:


df['Area_Name'].value_counts().head(10)


# Most Common Charges

# In[62]:


df['Charge_Group_Description'].value_counts().head(10)


# # Outlier Detection

# In[63]:


df['Age'].describe()


# In[64]:


Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower) | (df['Age'] > upper)]

print("Number of outliers:", len(outliers))


# # Missing Value Verification

# In[65]:


df.isnull().sum()


# 1. Cross_Street
# 
# Over 43,000 values are missing. This is likely because many arrest records simply do not have a cross street recorded.
# 
# You don't need this column for your EDA, so it's acceptable to leave it as is or even drop it if you don't plan to use it.

# In[66]:


df = df.drop(columns=['Cross_Street'])


# 2. Booking_Date, Booking_Time, Booking_Location_Code
# 
# These fields are missing together because not every arrest results in a bookings.
# 
# Therefore, do not fill these with fake values. Leaving them as NaN is appropriate

# 3. Arrest_Type_Code
# 
# There is only 1 missing value. Fill it with the most frequent value (mode):

# In[67]:


df['Arrest_Type_Code'].fillna(df['Arrest_Type_Code'].mode()[0], inplace=True)


# In[69]:


df.drop(columns=['Charge_Group_Code'], inplace=True)


# Since you'll likely use Charge_Group_Description instead of Charge_Group_Code for your analysis and visualizations
# Drop the column if you don't use it.

# During preprocessing, missing values in important categorical fields (such as Sex Code, Charge Description, and Disposition Description) were replaced with "Unknown". Columns such as Booking Date, Booking Time, and Booking Location Code were intentionally left with missing values because they are only applicable to records involving a booking. Similarly, Cross Street contained a large proportion of missing values and was excluded from further analysis since it was not essential to the project objectives.

# In[70]:


df.isnull().sum()


# # Save the cleaned dataset

# In[ ]:


# df.to_csv("Cleaned_Crime_Data.csv", index=False)

print("Cleaned dataset saved successfully.")


# # Summary of Data Preprocessing
# 
# 

# Removed duplicate records.
# Converted date columns to datetime format.
# Handled missing values in both categorical and numerical columns.
# Renamed column names for easier access.
# Removed redundant columns.
# Created new temporal features such as Arrest Year, Month, Day, and Hour.
# Saved the cleaned dataset for further analysis.

# # Visualization

# # 1. Age Distribution (Histogram)

# In[72]:


plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=25, kde=True)

plt.title("Distribution of Arrested Individuals by Age")
plt.xlabel("Age")
plt.ylabel("Number of Arrests")
plt.show()


# Insight
# 
# Most arrested individuals belong to the young and middle-aged population, while arrests involving elderly individuals are relatively uncommon.

# # 2. Arrests by Gender (Bar Chart)

# In[73]:


plt.figure(figsize=(6,5))

sns.countplot(data=df, x='Sex_Code')

plt.title("Arrests by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# Insight
# 
# Male individuals account for the majority of arrests, indicating a significant gender imbalance in arrest records.

# # 3. Top 10 Areas with Highest Arrests

# In[74]:


top_area = df['Area_Name'].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(x=top_area.values,
            y=top_area.index)

plt.title("Top 10 Areas with Highest Arrests")
plt.xlabel("Number of Arrests")
plt.ylabel("Area")
plt.show()


# Insight
# 
# A few areas have much higher arrest counts than the rest, suggesting these locations may experience more criminal activity or stronger policing efforts

# # 4. Monthly Arrest Trend (Line Chart)

# In[76]:


monthly = df.groupby('Arrest_Month').size()

plt.figure(figsize=(10,5))

plt.plot(monthly.index,
         monthly.values,
         marker='o')

plt.title("Monthly Arrest Trend")
plt.xlabel("Month")
plt.ylabel("Number of Arrests")

plt.xticks(rotation=45)

plt.show()


# Insight
# 
# Arrests were highest during August and September, while February recorded comparatively fewer arrests. This suggests that arrest activity changes throughout the year rather than remaining constant.

# # 5. Arrests by Hour

# In[77]:


hourly = df.groupby('Arrest_Hour').size()

plt.figure(figsize=(10,5))

sns.lineplot(x=hourly.index,
             y=hourly.values,
             marker='o')

plt.title("Arrests by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Number of Arrests")

plt.show()


# Insight
# 
# Arrest frequency varies throughout the day, helping identify peak hours of police activity.

# # 6. Top 10 Charge Categories

# In[78]:


charges = df['Charge_Group_Description'].value_counts().head(10)

plt.figure(figsize=(11,6))

sns.barplot(
    x=charges.values,
    y=charges.index
)

plt.title("Top 10 Charge Categories")
plt.xlabel("Number of Arrests")
plt.ylabel("Charge Category")

plt.show()


# Insight
# 
# A small number of charge categories account for a large proportion of arrests, with miscellaneous violations and assault-related offenses among the most common.

# # 7. Boxplot of Age by Gender

# In[79]:


plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x='Sex_Code',
    y='Age'
)

plt.title("Age Distribution by Gender")

plt.show()


# Insight
# 
# The age distribution differs slightly across genders, and the boxplot highlights the presence of outliers among older individuals.

# # 8. Correlation Heatmap

# In[80]:


num_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(8,6))

sns.heatmap(
    num_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()


# Insight
# 
# Most numerical variables show weak correlations, suggesting that factors such as age, reporting district, and geographic coordinates are largely independent.

# # 9. Arrest Locations (Scatter Plot)

# In[81]:


plt.figure(figsize=(8,6))

plt.scatter(
    df['LON'],
    df['LAT'],
    alpha=0.3,
    s=5
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("Geographic Distribution of Arrests")

plt.show()


# Insight
# 
# Arrests are concentrated in specific geographic regions rather than being uniformly distributed across the city.

# # 10. Arrest Type Distribution (Pie Chart)

# In[88]:


arrest_type = df['Arrest_Type_Code'].value_counts()

fig, ax = plt.subplots(figsize=(8,8))

wedges, texts, autotexts = ax.pie(
    arrest_type,
    labels=arrest_type.index,
    autopct='%1.1f%%',
    startangle=90
)

# Move the third percentage slightly downward
x, y = autotexts[2].get_position()
autotexts[2].set_position((x, y - 0.08))

plt.title("Distribution of Arrest Types")
plt.show()


# Insight
# 
# One arrest type constitutes the majority of records, indicating that certain law enforcement procedures occur much more frequently than others.

# # 11. Top 10 Areas by Average Age

# In[89]:


avg_age = (
    df.groupby('Area_Name')['Age']
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=avg_age.values,
    y=avg_age.index
)

plt.title("Top 10 Areas with Highest Average Age")
plt.xlabel("Average Age")
plt.ylabel("Area")

plt.show()


# Insight
# 
# Some police divisions report a higher average age among arrested individuals, suggesting differences in the demographic profile of arrests across areas.

# # Project insights
# 
# 

# 1. Male individuals account for the majority of arrest records, indicating a clear gender imbalance.
# 2. A small number of police divisions contribute a disproportionately high share of arrests, suggesting localized crime hotspots or greater enforcement.
# 3. Arrests peak during specific hours of the day, revealing temporal patterns in law enforcement activity.
# 4. Miscellaneous violations and assault-related offenses are among the most frequent charge categories, making them key contributors to overall arrest volume.
# 5. Arrest records are geographically clustered rather than evenly distributed, indicating that criminal activity and enforcement are concentrated in specific locations.
# 

# In[ ]:




