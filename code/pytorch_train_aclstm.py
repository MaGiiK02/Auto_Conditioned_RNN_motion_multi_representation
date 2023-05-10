import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import torch.nn.functional as tf
from utils.selectors import getHipAndRotationsFromSequence
import utils.losses as losses
import pandas as pd

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3
loss_record = pd.DataFrame(columns=['epoch', 'loss'])


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
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(seq_len):
            
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
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


#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, loss_function, iteration, frame_size, jont_data_size, data_rep, save_dance_folder, print_loss=False, save_bvh_motion=True, loss_record = pd.DataFrame(columns=['epoch', 'loss'])):
    Hip_index = read_bvh.joint_index['hip']
    if representation != "positional": Hip_index = 0 # Hip placed at the start of the data for rotations
    #set hip_x and hip_z as the difference from the future frame to current frame 
    dif = real_seq_np[:, 1:real_seq_np.shape[1]] - real_seq_np[:, 0: real_seq_np.shape[1]-1]
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
    real_seq  = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    seq_len=real_seq.size()[1]-1
    in_real_seq=real_seq[:, 0:seq_len]
    
    
    predict_groundtruth_seq= torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np[:,1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0],-1)
    
    
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)
    
    optimizer.zero_grad()
    if representation == "positional":
        loss=model.calculate_loss(predict_seq, predict_groundtruth_seq)
    else:
        # extract pos for hip pos error
        gt_rot, gt_pos = getHipAndRotationsFromSequence(predict_groundtruth_seq, jont_data_size, Hip_index, seq_len)
        pred_rot, pred_pos = getHipAndRotationsFromSequence(predict_seq, jont_data_size, Hip_index, seq_len)

        # cal error basen on positions and rotations
        pos_loss=model.calculate_loss(pred_pos, gt_pos)
        rot_loss=loss_function(pred_rot, gt_rot)

        #final Loss
        loss = rot_loss + pos_loss

    loss.backward()

    optimizer.step()
    
    if(print_loss==True):
        print ("###########"+"iter %07d"%iteration +"######################")
        print ("loss: "+str(loss))

    # add losses to array
    new_row = {'epoch':iteration, 'loss': loss.item()}
    loss_record = loss_record.append(new_row, ignore_index=True)
    
    if(save_bvh_motion==True):
        ##save the first motion sequence int the batch.
        gt_seq=np.array(predict_groundtruth_seq[0].data.tolist()).reshape(-1,frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(gt_seq.shape[0]):
            gt_seq[frame,Hip_index*3]=gt_seq[frame,Hip_index*3]+last_x
            last_x=gt_seq[frame,Hip_index*3]
            
            gt_seq[frame,Hip_index*3+2]=gt_seq[frame,Hip_index*3+2]+last_z
            last_z=gt_seq[frame,Hip_index*3+2]
        
        out_seq=np.array(predict_seq[0].data.tolist()).reshape(-1,frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_gt.bvh", gt_seq, data_rep)
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_out.bvh", out_seq, data_rep)

    return loss_record

#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=len(dance)/100
        length = 10
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
    
# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder, write_bvh_motion_folder, representation, lr=0.0001, total_iter=500000):
    loss_record = pd.DataFrame(columns=['epoch', 'loss'])
    seq_len=seq_len+2
    torch.cuda.set_device(0)

    #Extract details from representation
    jont_data_size = 3
    if representation == '6d': jont_data_size = 6
    elif representation == 'quaternions': jont_data_size = 4


    frame_size = 171 if representation == 'positional' else 3 + Joints_num*jont_data_size
    model = acLSTM(
        in_frame_size=frame_size,
        out_frame_size=frame_size
    )
        
    loss_function = model.calculate_loss
    if representation == 'euler': loss_function = losses.angle_distance
    elif representation == '6d': loss_function = losses.mae
    elif representation == 'quaternions': loss_function = losses.quaternion_loss    
    
    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path))
    
    model.cuda()
    #model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    
    model.train()
    
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    for iteration in range(total_iter):   
        #get a batch of dances
        dance_batch=[]
        for b in range(batch):
            #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0,random_range)]
            dance=dances[dance_id].copy()
            dance_len = dance.shape[0]
            
            #the first and last several frames are sometimes noisy. 
            start_id=random.randint(10, dance_len-seq_len*speed-10)
            sample_seq=[]
            for i in range(seq_len):
                sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
            
            #augment the direction and position of the dance
            if representation == 'positional':
                # Noise Injection
                T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
                R=[0,1,0,(random.random()-0.5)*np.pi*2]
                sample_seq_augmented=read_bvh.augment_positional(sample_seq, T, R)
                dance_batch=dance_batch+[sample_seq_augmented]
            else:
                dance_batch += [sample_seq]
            
        dance_batch_np=np.array(dance_batch)
       
        
        print_loss=False
        save_bvh_motion=False
        if(iteration % 1==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            
        loss_record = train_one_iteraton(dance_batch_np, model, optimizer, loss_function, iteration, frame_size, jont_data_size, representation, write_bvh_motion_folder, print_loss, save_bvh_motion, loss_record)
        #end=time.time()
        #print end-start
        if(iteration%1000 == 0):
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)
            
        path = os.path.dirname(os.path.dirname(write_weight_folder)) + "/loss.csv"
        loss_record.to_csv(path)

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    dance_files = [d for d in dance_files if ".bvh.npy" in d] # Pick only corret files types
    for dance_file in dance_files:
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file, allow_pickle=True)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance]
    return dances

if __name__ == '__main__' :
    # Eample of to launch via comandline
    # python ./pytorch_train_aclsym --representation=positional --dances_folder=train_data_xyz/positional/ --write_weight_folder=run/postitional/weigths/ --write_bvh_motion_folder=runs/positional/bvh/ --dance_frame_rate=120 --batch_size=32 --epochs=50000
    parser = argparse.ArgumentParser(description='ACLSTM-Train')
    parser.add_argument('--read_weight_path', default='',
                help='Path where to load the weights.')
    parser.add_argument('--write_weight_folder', default=None,
                help='Output folder where to store the weights.')
    parser.add_argument('--write_bvh_motion_folder', default=None,
                help='Path to the whwere tostore the output bvh files.')
    parser.add_argument('--dances_folder', default=None,
                help='Path to the folder contining the original bvh files.')
    parser.add_argument('--dance_frame_rate', default=None,
                help='The framerate of the bvh files.')
    parser.add_argument('--batch_size', default=32,
                help='Train batches size (Default: 32)')
    parser.add_argument('--epochs', default=10,
                help='Train batches size (Default: 10) Note Paper recomends 500000')
    parser.add_argument('--lr', default=0.0001,
                help='The learning rate to use. (Default: 0.0001)')
    parser.add_argument('--representation', default=None,
            help='The representation to use to represent the rotation to the model [positional, euler, 6d, quaternions], used to infer the loss function.')
    
    args = parser.parse_args()

    read_weight_path=args.read_weight_path                   # Example None
    write_weight_folder=args.write_weight_folder             # Example "../train_weight_aclstm_indian/"
    write_bvh_motion_folder=args .write_bvh_motion_folder    # Example "../train_tmp_bvh_aclstm_indian/"
    dances_folder=args.dances_folder                         # Example "../train_data_xyz/indian/"
    dance_frame_rate=int(args.dance_frame_rate)              # Example 60
    batch=int(args.batch_size)                               # Example 32 
    epochs=int(args.epochs)                                  # Example 2000000 
    representation=args.representation
    lr = float(args.lr)                                      # Example 0.0001 
                                    

    if not os.path.exists(write_weight_folder):
        os.makedirs(write_weight_folder)
    if not os.path.exists(write_bvh_motion_folder):
        os.makedirs(write_bvh_motion_folder)

    gen_info_path = f'{os.path.dirname(write_weight_folder)}/info.txt'
    os.makedirs(os.path.dirname(write_weight_folder), exist_ok=True)
    with open(gen_info_path, "w") as f:
        f.write(f'{args}')    

    dances = load_dances(dances_folder)

    ## representations switch
    train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder, write_bvh_motion_folder, representation, lr, epochs)


