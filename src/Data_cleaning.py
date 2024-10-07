import os
import argparse
import pandas as pd
import mlflow

raw_data_path = os.path.abspath("./../artifacts/data/raw_data/")
clean_data_path= os.path.abspath("./../artifacts/data/cleaned_data/")
#df = pd.read_csv(raw_data_path)
raw_data_file= "homeprices.csv"
#print(raw_data_path)
#print(processed_data_path)

def genarate_file_name(file_name):
    reviewed_file_name = file_name.split(".csv")
    reviewed_file_name.insert(-1, "clean.csv")
    reviewed_file_name.pop()
    reviewed_file_name= "_".join(reviewed_file_name)
    return reviewed_file_name

def data_cleaning(raw_data_path, clean_data_path, raw_data_file):
    raw_data= os.path.join(raw_data_path, raw_data_file)
    df = pd.read_csv(raw_data)

    clean_data_file = genarate_file_name(raw_data_file)
    clean_data = os.path.join(clean_data_path, clean_data_file)
    mlflow.log_param("clean_data_path",clean_data)
    print(f"exposed clean data {clean_data}")
    df.to_csv(clean_data, index=False)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", help="provide raw data", default=raw_data_path)
    parser.add_argument("--clean_data_path", help="provide clean data", default=clean_data_path)
    parser.add_argument("--raw_data_file", help="provide raw file", default=raw_data_file)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path, args.clean_data_path, args.raw_data_file)
   # print(args)
