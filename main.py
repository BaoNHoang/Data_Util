import pandas as pd
from data_utils import load_and_format_csv, scale_dataset

def main():
    
    csv_path = "patients.csv"  
    df = load_and_format_csv(csv_path)

    print("\n--- Raw Data (Cleaned) ---")
    print(df.head())

    data, X, y = scale_dataset(df, oversample=True)

    print("\n--- Scaled + Oversampled Data ---")
    print(data[:5])  
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    output_df = pd.DataFrame(data, columns=list(df.columns))
    output_df.to_csv("patients_cleaned.csv")
    print("\nSaved cleaned dataset to 'patients_cleaned.csv'")

if __name__ == "__main__":
    main()
