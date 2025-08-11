import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

num_accounts = 200
num_transactions = 1000

data = []

# Parameters
legit_ratio = 0.8
fraud_ratio = 0.15
group_ratio = 0.05

# Legit transactions
for _ in range(int(num_transactions * legit_ratio)):
    src = random.randint(0, num_accounts - 1)
    dst = random.randint(0, num_accounts - 1)
    while dst == src:
        dst = random.randint(0, num_accounts - 1)
    amount = round(random.uniform(50, 5000), 2)
    data.append([src, dst, amount, 0])

# Single fraud transactions
for _ in range(int(num_transactions * fraud_ratio)):
    src = random.randint(0, num_accounts - 1)
    dst = random.randint(0, num_accounts - 1)
    while dst == src:
        dst = random.randint(0, num_accounts - 1)
    amount = round(random.uniform(7000, 200000), 2)
    data.append([src, dst, amount, 1])

# Group laundering (high-value ring)
num_groups = int(num_transactions * group_ratio // 4)  # each group ~4 transactions
for _ in range(num_groups):
    group_accounts = random.sample(range(num_accounts), 4)
    for i in range(4):
        src = group_accounts[i]
        dst = group_accounts[(i + 1) % 4]
        amount = round(random.uniform(100000, 200000), 2)
        data.append([src, dst, amount, 2])

# Shuffle dataset
random.shuffle(data)

# Save
df = pd.DataFrame(data, columns=["source", "target", "amount", "label"])
df.to_csv("data/transactions01.csv", index=False)

print("✅ transactions.csv generated with shape:", df.shape)
print(df.head(10))
