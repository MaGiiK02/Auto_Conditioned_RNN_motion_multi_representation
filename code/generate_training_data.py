import read_bvh
import numpy as np
from os import listdir
import argparse
import os


def generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation):
    print ("Generating training data for "+ src_bvh_folder)
    if (os.path.exists(tar_traindata_folder)==False):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names=listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len=len(bvh_dance_name)
        if(name_len>4):
            if(bvh_dance_name[name_len-4: name_len]==".bvh"):
                print ("Processing "+bvh_dance_name)
                dance=read_bvh.get_train_data(src_bvh_folder+bvh_dance_name, representation)
                np.save(tar_traindata_folder+bvh_dance_name+".npy", dance)
                
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='ACLSTM-Train')
    parser.add_argument('in_folder', default=None, help='Path to the folder containig the bvh to be processed.')
    parser.add_argument('out_folder', default=None, help='Path to the folder where to output theprocessed bvh.')
    parser.add_argument('--representation', default=None, help='Which representation to use for the conversion. [positional, euler, 6d, quaternions]')
    
    args = parser.parse_args()

    gen_info_path = f'{args.out_folder}/info.txt'
    os.makedirs(args.out_folder, exist_ok=True)
    with open(gen_info_path, "w") as f:
        f.write(f'{args}')

    generate_traindata_from_bvh(args.in_folder, args.out_folder, args.representation)
    #generate_traindata_from_bvh("../train_data_bvh/indian/","../train_data_xyz/indian/")
    #generate_traindata_from_bvh("../train_data_bvh/salsa/","../train_data_xyz/salsa/")
    #generate_traindata_from_bvh("../train_data_bvh/martial/","../train_data_xyz/martial/")