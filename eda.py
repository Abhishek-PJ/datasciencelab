import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("automobile_data2.csv")

# Dimension
#print("Shape of the dataset:", df.shape)

# Structure
#print("\nColumns and Data Types:\n", df.dtypes)

# Summary
#print("\nSummary:\n", df.describe(include='all'))

#data preprocessing

# Check for missing values
#print(df.isnull().sum())  # Shows count of missing values in each column
# Fill numeric columns with the mean
# df.fillna(df.mean(numeric_only=True), inplace=True)
# df.to_csv("automobile_data2.csv", index=False)


# Display the cleaned dataset
#print(df.describe)


#Histogram

# fig,axes=plt.subplots(1,2,figsize=(10,10))

# # Plotting the histogram for Length on the second subplot
# axes[0].hist(df["price"], bins=10, color="blue", edgecolor="black")
# axes[0].set_title("Price")  # Set title for the first subplot
# axes[0].set_xlabel("Price")
# axes[0].set_ylabel("Frequency")

# # Plotting the histogram for Length on the second subplot
# axes[1].hist(df["length"], bins=10, color="blue", edgecolor="black")
# axes[1].set_title("Length")  # Set title for the second subplot
# axes[1].set_xlabel("Length")
# axes[1].set_ylabel("Frequency")

# plt.show()

#violinplot

# plt.figure(figsize=(8, 6))
# sns.violinplot(x='fuel-type', y='price', data=df)
# plt.title('Price Distribution by Fuel Type')
# plt.show()


#boxplot

# Display box plot before outlier treatment
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['price'])
plt.title('Box Plot Before Outlier Treatment')
plt.show()

# Identify outliers using IQR (Interquartile Range)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data to remove outliers
df_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Display box plot after outlier treatment
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_filtered['price'])
plt.title('Box Plot After Outlier Treatment')
plt.show()
