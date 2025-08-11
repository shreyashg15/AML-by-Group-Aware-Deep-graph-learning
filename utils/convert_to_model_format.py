import pandas as pd

def convert_fraud_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # 🔍 Auto-detect column mappings
    if 'nameOrig' in df.columns and 'nameDest' in df.columns:
        src_col, dst_col = 'nameOrig', 'nameDest'
    elif 'sender' in df.columns and 'receiver' in df.columns:
        src_col, dst_col = 'sender', 'receiver'
    elif 'source' in df.columns and 'target' in df.columns:
        print("⚠️ Dataset already in converted format. Copying.")
        df[['source', 'target', 'amount', 'label']].to_csv(output_csv, index=False)
        return
    else:
        raise ValueError("❌ Unknown column format. Must include nameOrig/nameDest or sender/receiver.")

    # ⚙️ Set amount and label
    if 'amount' not in df.columns:
        raise ValueError("❌ Missing 'amount' column.")
    
    if 'isFraud' in df.columns:
        label_col = 'isFraud'
    elif 'is_suspicious' in df.columns:
        label_col = 'is_suspicious'
    else:
        raise ValueError("❌ Missing fraud label column (isFraud or is_suspicious)")

    # ✅ Build output DataFrame
    df_model = pd.DataFrame({
        'source': df[src_col],
        'target': df[dst_col],
        'amount': df['amount'],
        'label': df[label_col]
    })

    df_model.to_csv(output_csv, index=False)
    print(f"✅ Converted and saved to: {output_csv}")
