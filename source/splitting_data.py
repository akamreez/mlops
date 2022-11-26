import os
import yaml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from getdata import GetData

class SplittingData(object):
    def __init__(self) :
        pass
    def train_test_split_and_save_data(self,config_path):
        self.config=GetData().read_params(config_path)
        self.raw_data_path=self.config["Data"]["data_raw"]
        self.train_data=self.config["split_data"]["train_path"]
        self.test_data=self.config["split_data"]["test_path"]
        self.split_ratio=self.config["split_data"]["test_size"]
        self.data=pd.read_csv(self.raw_data_path,sep=",")
        self.random_state=self.config["base"]["random-state"]
        self.train,self.test=train_test_split(self.data,test_size=self.split_ratio,random_state=self.random_state)
        self.dir="data/processed"
        self.make_dir=os.path.join(os.getcwd(),self.dir)
        os.makedirs(self.make_dir,exist_ok=True)
        self.train.to_csv(self.train_data,sep=",",index=False,encoding="UTF-8")
        self.test.to_csv(self.test_data,sep=",",index=False,encoding="UTF-8")
        print("successfully executed")
        return self.train,self.test

if __name__=="__main__":
    args=argparse.ArgumentParser(prog="load_data",description="This is for loading my data to pipeline")
    args.add_argument("--config",default="params.yaml")
    parse_args=args.parse_args()
    get_data=SplittingData().train_test_split_and_save_data(config_path=parse_args.config)
