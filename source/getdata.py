import os
import yaml
import pandas as pd
import numpy as np
import argparse

class GetData(object):
    def __init__(self) :
        pass
    def read_params(self,config_path):
        with open(config_path) as yaml_file:
            config=yaml.safe_load(yaml_file)
        return config
    

    def get_data(self,config_path):
        self.config=self.read_params(config_path)
        self.data=self.config["Data"]["data_raw"]
        self.data_read=pd.read_csv(self.data,sep=",")
        print(self.data_read)
        return self.data_read
    
#GetData().get_data("params.yaml")
if __name__=="__main__":
    args=argparse.ArgumentParser(prog="get_data",description="This is for getting my data",epilog="Something")
    args.add_argument("--config",default="params.yaml")
    parse_args=args.parse_args()
    get_data=GetData().get_data(config_path=parse_args.config)
    



        
    