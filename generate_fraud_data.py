import pandas as pd
import random

def generate_aml_format_data(num_groups=5, users_per_group=5, normal_txns=975):
    data = []
    step = 1
    user_id = 1000

    for g in range(num_groups):
        users = [f'G{g}_U{i}' for i in range(users_per_group)]
        for i in range(len(users) - 1):
            nameOrig = users[i]
            nameDest = users[i + 1]
            amount = round(random.uniform(9000, 20000), 2)
            oldbalanceOrg = round(random.uniform(5000, 15000), 2)
            newbalanceOrig = round(oldbalanceOrg - amount, 2)
            data.append([
                step, "TRANSFER", amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                nameDest, 0.0, 0.0, 1, 0
            ])
            step += 1
        amount = round(random.uniform(10000, 25000), 2)
        data.append([
            step, "TRANSFER", amount, users[-1], 12000.0, round(12000.0 - amount, 2),
            users[0], 0.0, 0.0, 1, 0
        ])
        step += 1

    for i in range(normal_txns):
        nameOrig = f'C{user_id + i}'
        nameDest = f'M{user_id + i + 1}'
        amount = round(random.uniform(100, 5000), 2)
        oldbalanceOrg = round(random.uniform(1000, 10000), 2)
        newbalanceOrig = round(oldbalanceOrg - amount, 2)
        data.append([
            step, "PAYMENT", amount, nameOrig, oldbalanceOrg, newbalanceOrig,
            nameDest, 0.0, 0.0, 0, 0
        ])
        step += 1

    df = pd.DataFrame(data, columns=[
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
    ])
    df.to_csv("high_fraud_aml_dataset.csv", index=False)
    print("✅ File saved: high_fraud_aml_dataset.csv")

generate_aml_format_data()
