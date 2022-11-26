import os
import yaml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from getdata import GetData
from urllib.parse import urlparse
import joblib
import json
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
class TrainingPipeline(object):
    def __init__(self):
        pass
    
    def eval_metrics(self,actual,predicted):
        self.rmse=np.sqrt(mean_squared_error(actual,predicted))
        self.mae=mean_squared_error(actual,predicted)
        self.r2_score=r2_score(actual,predicted)
        return self.r2_score,self.mae,self.rmse
    def model_evaluation(self,config_path):
        self.config=GetData().read_params(config_path)
        self.test_data=self.config["split_data"]["test_path"]
        self.train_data=self.config["split_data"]["train_path"]
        self.model_dir=self.config["model_dir_save_pkl"]
        self.alpha=self.config["estimators"]["ElasticNet"]["params"]["alpha"]
        self.l1_ratio=self.config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
        self.target=self.config["base"]["target_col"] 
        self.random_state=self.config["base"]["random-state"]
        self.train=pd.read_csv(self.train_data,sep=",")
        self.test=pd.read_csv(self.test_data,sep=",")
        print(self.test.columns)
        self.y_train=self.train[self.target]
        self.y_test=self.test[self.target]
        self.x_train=self.train.drop(self.target,axis=1)
        self.x_test=self.test.drop(self.target,axis=1)
        self.El=ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,random_state=self.random_state)
        self.El.fit(self.x_train,self.y_train)
        print("fitted my data with elasticnet")
        self.y_pred=self.El.predict(self.x_test)
        (self.r2_score,self.mae,self.rmse)=TrainingPipeline().eval_metrics(self.y_test,self.y_pred)
        self.dir="elasticnet_model_dir"
        self.model_dir=os.path.join(os.getcwd(),self.dir)
        os.makedirs(self.model_dir,exist_ok=True)
        self.model_dir_file=os.path.join(self.model_dir,"elasticnet_model_binary.pkl")
        joblib.dump(self.El,self.model_dir_file)
        self.scores_model_dir="reports"
        os.makedirs(self.scores_model_dir,exist_ok=True)
        self.score_file=self.config["reports"]["scores"]
        self.params_file=self.config["reports"]["params"]
        with open(self.score_file,"w+") as score_file:
            score={"rmse":self.rmse,"mae":self.mae,"r2_score":self.r2_score}
            json.dump(score,score_file,indent=5)
        #parameters of my model
        with open(self.params_file,"w+") as params_file:
            params={"alpha":self.alpha,"l1_ratio":self.l1_ratio}
            json.dump(params,params_file,indent=5)


if __name__=="__main__":
    args=argparse.ArgumentParser(prog="load_data",description="This is for loading my data to pipeline")
    args.add_argument("--config",default="params.yaml")
    parse_args=args.parse_args()
    get_data=TrainingPipeline().model_evaluation(config_path=parse_args.config)

