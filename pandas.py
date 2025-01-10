Creating a DataFrame from a Dictionary:
import pandas as pd
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df)

Output:

    Name    Age
0   Alice   25
1   Bob     30
-----------------------------------------------
Adding a New Column:
df['City'] = ['New York', 'Los Angeles']
print(df)

Output:
     Name   Age         City
0    Alice  25       New York
1     Bob   30      Los Angeles
-----------------------------------------------
Handling Missing Data:
data = {'Name': ['Alice', 'Bob', None], 'Age': [25, None, 30]}
df = pd.DataFrame(data)
filled = df.fillna('Unknown')
print(filled)

 Output:
   Name         Age
0  Alice        25.0
1  Bob        Unknown
2  Unknown      30.0
-----------------------------------------------
import pandas as pd
data = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'}
]
df = pd.DataFrame(data)
print(df)



Dropping Columns or Rows:
df = df.drop(columns=['City'])  # Drop the 'City' column
print(df)

 Output:
     FullName  Years
0      Alice     25
1        Bob     30
-----------------------------------------------

Checking for Missing Data:
df = pd.DataFrame({'Name': ['Alice', 'Bob', None], 'Age': [25, None, 30]})
print(df.isnull())  # Check for missing values

 Output:
    Name    Age
0  False  False
1  False   True
2   True  False

-----------------------------------------------
Dropping Rows with Missing Data:
df = pd.DataFrame({'Name': ['Alice', 'Bob', None], 'Age': [25, None, 30]})
df_cleaned = df.dropna()
print(df_cleaned)

 Output:
     Name     Age
0    Alice    25.0
2    Bob      30.0
-----------------------------------------------

Merging DataFrames:
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Sales': [250, 300]})
merged = pd.merge(df1, df2, on='ID')
print(merged)

 Output:
   ID   Name  Sales
0   1  Alice    250
1   2    Bob    300


