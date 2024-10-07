import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import pickle
import os
import argparse

# model =pickle.load(open("./../artifacts/model/model.pkl", "rb"))
# x_test = pd.read_csv("./../artifacts/data/processed/x_test.csv")
# y_test = pd.read_csv("./../artifacts/data/processed/y_test.csv")
processed_data_path = os.path.abspath("./../artifacts/data/processed/")
model_path = os.path.abspath("./../artifacts/model/")
eval_model_path = os.path.abspath("./../artifacts/model_eval")

#  y_pred_test= model.predict(x_test)


# r2_score=r2_score(y_test, y_pred_test)
# MSE=mean_squared_error(y_test, y_pred_test)
# MAE=mean_absolute_error(y_test, y_pred_test)

# if r2_score > 0.8:
#     print(r2_score)
#     pickle.dump(model,open("./../artifacts/model.eval", "wb"))
    
def eval_model(processed_data_path, model_path, eval_model_path):
    model_path_file = os.path.join(model_path, "my_model.pkl")
    model = pickle.load(open(model_path_file, "rb"))

    x_test_path = os.path.join(processed_data_path, "x_test.csv")
    y_test_path = os.path.join(processed_data_path, "y_test.csv")

    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    
    y_test_pred = model.predict(x_test)

    r_score = r2_score(y_test, y_test_pred)
    MSE = mean_squared_error(y_test, y_test_pred)
    MAE = mean_absolute_error(y_test, y_test_pred)

    if r_score > 0.8:
        print(r_score)
        eval_model_path_file = os.path.join(eval_model_path, "eval_model.pkl")
        pickle.dump(model, open(eval_model_path_file, "wb"))
    else:
        print(f"[WARNING] model doesnt pass evalution criteria r_score = {r_score}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path", help="provide processed data", default=processed_data_path)
    parser.add_argument("--model_path", help="provide model data", default=model_path)
    parser.add_argument("--eval_model_path", help="provide eval model path", default=eval_model_path)
    args = parser.parse_args()
    eval_model(args.processed_data_path, args.model_path, args.eval_model_path)




   
   
 
 
 
