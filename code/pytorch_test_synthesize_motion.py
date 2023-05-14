import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import pandas as pd
from utils.forward_kinematics import fk_6D,fk_euler,fk_quaternions

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Rotational_joints_num = len(read_bvh.rotational_joints_index)
Condition_num=5
Groundtruth_num=5


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        
        batch=initial_seq.size()[0]
        
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
        
        for i in range(generate_frames_number):
            
            in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


#numpy array inital_seq_np: batch*seq_len*frame_size
#return numpy b*generate_frames_number*frame_data
def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder, representation,In_frame_size):
    Hip_index = read_bvh.joint_index['hip']
    if representation != "positional": Hip_index = 0 # Hip placed at the start of the data for rotations
    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
    initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
    initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    predict_seq = model.forward(initial_seq, generate_frames_number)
    
    batch=initial_seq_np.shape[0]
   
    for b in range(batch):
        
        out_seq=np.array(predict_seq[b].data.tolist()).reshape(-1,In_frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq, representation)
    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)



#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        length=len(dance)/100
        length=10
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    for dance_file in dance_files:
        if not ".bvh" in dance_file: continue
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance]
    return dances
    
# dances: [dance1, dance2, dance3,....]
def test(dance_batch_np, frame_rate, dances_test_size, initial_seq_len, generate_frames_number, read_weight_path, write_bvh_motion_folder, representation):
    
    torch.cuda.set_device(0)

    jont_data_size = 3
    if representation == '6d': jont_data_size = 6
    elif representation == 'quaternions': jont_data_size = 4

    frame_size = Joints_num*3 if representation == 'positional' else 3 + Rotational_joints_num*jont_data_size
    model = acLSTM(
        in_frame_size=frame_size,
        out_frame_size=frame_size
    )  
    
    model.load_state_dict(torch.load(read_weight_path))
    
    model.cuda()
    
    # Gen random seed to enable loading the same dances for different testes
    seeded_random =np.random.RandomState(seed=0)
    
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    
    dance_batch=[]
    for b in range(dances_test_size):
        #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
        dance_id = dance_len_lst[seeded_random.randint(0,random_range)]
        dance=dances[dance_id].copy()
        dance_len = dance.shape[0]
            
        start_id= seeded_random.randint(10, int(dance_len-initial_seq_len*speed-10)) #the first and last several frames are sometimes noisy. 
        sample_seq=[]
        for i in range(initial_seq_len):
            sample_seq=sample_seq+[dance[int(i*speed+start_id)]]

        dance_batch=dance_batch+[sample_seq]

   

    # Prediction        
    dance_batch_np=np.array(dance_batch)    
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder, representation, frame_size)
   


if __name__ == '__main__' :
    # Eample of to launch via comandline
    # python ./pytorch_train_aclsym --representation=positional --dances_folder=train_data_xyz/positional/ --write_weight_folder=run/postitional/weigths/ --write_bvh_motion_folder=runs/positional/bvh/ --dance_frame_rate=120 --batch_size=32 --epochs=50000
    parser = argparse.ArgumentParser(description='ACLSTM-Test & Syntesis')
    parser.add_argument('--read_weight_path', default='',
                help='Path where to load the weights.')
    parser.add_argument('--write_bvh_motion_folder', default=None,
                help='Path to the whwere to store the output bvh files.')
    parser.add_argument('--dances_folder', default=None,
                help='Path to the folder contining the original bvh files.')
    parser.add_argument('--dance_frame_rate', default=None,
                help='The framerate of the bvh files.')
    parser.add_argument('--dances_test_size', default=5,
                help='The number of dances to use to do the evaluation (note :dances will be processed as a single batch as such keep the number low(0 Default: 5)')
    parser.add_argument('--initial_seq_len', default=15,
                help='The amount of frames to use as inputo to start the generation. (Default: 15)')
    parser.add_argument('--generate_frames_number', default=400,
                help='The amount of frames to generate to create the motion. (Default: 400)')
    parser.add_argument('--representation', default=None,
            help='The representation to use to represent the rotation to the model [positional, euler, 6d, quaternions], used to infer the loss function.')

    args = parser.parse_args()

    read_weight_path=args.read_weight_path                   # Example None           # Example "../train_weight_aclstm_indian/"
    write_bvh_motion_folder=args .write_bvh_motion_folder    # Example "../train_tmp_bvh_aclstm_indian/"
    dances_folder=args.dances_folder                         # Example "../train_data_xyz/indian/"
    dance_frame_rate=int(args.dance_frame_rate)              # Example 60
    dances_test_size=int(args.dances_test_size)              # Example 5                              
    representation=args.representation                       # Example positional 
    initial_seq_len=args.initial_seq_len                     # Example 15 
    generate_frames_number=args.generate_frames_number       # Example 400

    if not os.path.exists(write_bvh_motion_folder):
        os.makedirs(write_bvh_motion_folder)
        

dances= load_dances(dances_folder)

test(dances, dance_frame_rate, dances_test_size, initial_seq_len, generate_frames_number, read_weight_path,  write_bvh_motion_folder, representation)



