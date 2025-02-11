import json
import argparse
from utils import plot
from train import train
from shutil import copyfile
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--load_from', help='read from json file')
    parser.add_argument('--results_dir', help='where to save results')
    args = parser.parse_args()

    file = open(args.load_from+".json")
    jsonname = args.load_from.split("/")[-1]
    hyperparams = json.load(file)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    copyfile(args.load_from+".json",args.results_dir+"\\"+jsonname+".json")
    train(hyperparams = hyperparams, save_dir = args.results_dir)

    
    

