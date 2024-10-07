import pandas as pd 
import os
import mlflow
import argparse
from sklearn.model_selection import train_test_split

#df = pd.read_csv("./../artifacts/data/cleaned_data/homeprices_clean.csv")
cleaned_data_path = os.path.abspath("./../artifacts/data/cleaned_data")
processed_data_path = os.path.abspath("./../artifacts/data/processed")

def processed_data(cleaned_data_path, processed_data_path,Target):
    cleaned_data_file = os.listdir(cleaned_data_path)[0]
    cleaned_data = os.path.join(cleaned_data_path, cleaned_data_file)
    df = pd.read_csv(cleaned_data)
    Y = df[[Target]]
    X = df.drop(columns=[Target])
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.4)
    mlflow.log_param("test_size",0.4)
    
    x_train.to_csv(os.path.join(processed_data_path, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(processed_data_path, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_data_path, "y_test.csv"), index=False)
    
# x_train.to_csv("./../artifacts/data/processed/x_train.csv")
# x_test.to_csv("./../artifacts/data/processed/x_test.csv")
# y_train.to_csv("./../artifacts/data/processed/y_train.csv")
# y_test.to_csv("./../artifacts/data/processed/y_test.csv")
  

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path", help="provide clean data", default=cleaned_data_path)
    parser.add_argument("--processed_data_path", help="provide processed data", default=processed_data_path)
    parser.add_argument("--Target", help="provide Target variable", default=None)
    args = parser.parse_args()
    print(args)
    if args.Target !=None:
        processed_data(args.cleaned_data_path, args.processed_data_path, args.Target)
    else:
        print(f"[ERROR] Something went wrong. Target variable can't be None {args.Target}")