import os
import yaml
import pandas as pd
import numpy as np
import argparse
from getdata import GetData

class LoadData(object):
    def __init__(self) :
        pass
    def load_data(self,config_path):
        self.config=GetData.read_params(config_path)
        self.data=GetData.get_data(config_path)
        self.new_columns=[col.replace(" ","-") for col in self.data.columns]
        self.raw_data=self.config["Data"]["data_raw"]
        self.data.to_csv(self.raw_data,sep=",",index=False,header=self.new_columns)
        
        return self.data




if __name__=="__main__":
    args=argparse.ArgumentParser(prog="load_data",description="This is for loading my data",epilog="Something")
    args.add_argument("--config",default="params.yaml")
    parse_args=args.parse_args()
    get_data=GetData().get_data(config_path=parse_args.config)

        