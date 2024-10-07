import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import argparse
import os
import mlflow

processed_data_path = os.path.abspath("./../artifacts/data/processed/")
model_path = os.path.abspath("./../artifacts/model/")

# x_train = pd.read_csv("./../artifacts/data/processed/x_train.csv")
# y_train = pd.read_csv("./../artifacts/data/processed/y_train.csv")

#pickle.dump(model, open("./../artifacts/model/model.pkl", "wb"))

def model_building(processed_data_path, model_path):
    print("################MODEL BUILDING STARTED#####################")
    x_train_path = os.path.join(processed_data_path, "x_train.csv")
    y_train_path = os.path.join(processed_data_path, "y_train.csv")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    print("[INFO] Model building started")

    model = LinearRegression()
    model.fit(x_train, y_train)
    
    model_path_file = os.path.join(model_path, "my_model.pkl")
    pickle.dump(model, open(model_path_file, "wb"))
    print(f"[INFO] model is expossed to {model_path_file}")
    mlflow.log_param("model_path",model_path_file)
    mlflow.sklearn.log_model(model,"my-model")
    print("################MODEL BUILDING FINISHED#####################")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="provide processed data", default=processed_data_path)
    parser.add_argument("--model_path", help="provide model path", default=model_path)
    args = parser.parse_args()
    model_building(args.processed_data_path, args.model_path)




