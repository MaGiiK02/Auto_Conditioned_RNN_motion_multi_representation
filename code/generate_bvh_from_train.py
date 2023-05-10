import read_bvh
import numpy as np
from os import listdir
import argparse
import os

def generate_bvh_from_traindata(src_train_folder, tar_bvh_folder, representation):
    print ("Generating bvh data for "+ src_train_folder)
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)
    dances_names=listdir(src_train_folder)
    for dance_name in dances_names:
        name_len=len(dance_name)
        if(name_len>4):
            if(dance_name[name_len-4: name_len]==".npy"):
                print ("Processing"+dance_name)
                dance=np.load(src_train_folder+dance_name)
                dance2=[]
                for i in range(dance.shape[0]/8):
                    dance2=dance2+[dance[i*8]]
                print (len(dance2))
                read_bvh.write_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh",np.array(dance2), representation)
                
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='ACLSTM-Train')
    parser.add_argument('in_folder', default=None, help='Path to the folder containig the numpy file to be converted back')
    parser.add_argument('out_folder', default=None, help='Path to the folder where to output the reconstructed bvh.')
    parser.add_argument('--representation', default=None, help='Which representation to use for the conversion. [positional, euler, 6D, quaternions]')
    
    args = parser.parse_args()

    generate_bvh_from_traindata(args.in_folder, args.out_folder, args.representation)