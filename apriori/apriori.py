import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
dataset = pd.read_csv('basket.csv', header=None)

# Assuming each row represents a transaction and each item is separated by a comma
transactions = [row.str.split(',') for _, row in dataset.iterrows()]

# Use TransactionEncoder to one-hot encode the dataset
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.05  # Adjust the minimum support as needed
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 0.7  # Adjust the minimum confidence as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Print the frequent itemsets and association rules
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
