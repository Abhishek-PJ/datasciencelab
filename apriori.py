import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("./basket.csv")

# Fill missing values
df.fillna('', inplace=True)

# One-hot encoding
df_dum = pd.get_dummies(df)

# Apriori algorithm to find frequent itemsets
frequent_items = apriori(df_dum, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.01)

# Sort rules by support and confidence
rules = rules.sort_values(['support', 'confidence'], ascending=[False, False])

# Display results
print(rules)
