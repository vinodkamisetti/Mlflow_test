import argparse
import mlflow
import dagshub

dagshub.init(repo_owner='vinodkamisetti', repo_name='Mlflow_test', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/vinodkamisetti/Mlflow_test.git")
mlflow.set_experiment("homeprice_dagshub")

def main(Target):
    print(f"[INFO] MLOps Pipeline Triggerd for")
    with mlflow.start_run() as run:
        mlflow.run("./src", entry_point="Data_Cleaning.py",env_manager="local")
        #mlflow.run("./src",entry_point="Data_Preprocessing.py",parameters={'Target':Target},env_manager="local")
        mlflow.run("./src", entry_point="Data_Preprocessing.py", parameters={'Target': Target}, env_manager="local")
        mlflow.run("./src", entry_point="Model_Building.py", env_manager="local")
        mlflow.run("./src", entry_point="Model_Evaluation.py", env_manager="local")
        #mlflow.run("./src",entry_point="Model_Building.py")
        #mlflow.run("./src","entry_point="Model_Evaluaion.py", env_manager = "local")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--Target",type=str, default=None)
    parser.add_argument("--Target", help="provide Target variable", required=True)
    args = parser.parse_args()
    main(args.Target)



        