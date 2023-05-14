
import numpy as np
import cv2 as cv
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from collections import Counter

import transforms3d.euler as euler
import transforms3d.quaternions as quat
import torch
from pylab import *
from PIL import Image
import os
import getopt

import json # For formatted printing

import read_bvh_hierarchy

import rotation2xyz as helper
from rotation2xyz import *

import utils.rotations_conversion as rot_conv

def get_pos_joints_index(raw_frame_data, non_end_bones, skeleton):
    pos_dic=  helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    keys=OrderedDict()
    i=0
    for joint in pos_dic.keys():
        keys[joint]=i
        i=i+1
    return keys


def parse_frames(bvh_filename):
   bvh_file = open(bvh_filename, "r")
   lines = bvh_file.readlines()
   bvh_file.close()
   l = [lines.index(i) for i in lines if 'MOTION' in i]
   data_start=l[0]
   #data_start = lines.index('MOTION\n')
   first_frame  = data_start + 3
   
   num_params = len(lines[first_frame].split(' ')) 
   num_frames = len(lines) - first_frame
                                     
   data= np.zeros((num_frames,num_params))
   
   for i in range(num_frames):
       line = lines[first_frame + i].split(' ')
       line = line[0:len(line)]

       
       line_f = [float(e) for e in line]
       
       data[i,:] = line_f
           
   return data
   
def get_frame_format_string(bvh_filename):
    bvh_file = open(bvh_filename, "r")
    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if 'MOTION' in i]
    data_end=l[0]
    #data_end = lines.index('MOTION\n')
    data_end = data_end+2
    return lines[0:data_end+1]

def get_min_foot_and_hip_center(bvh_data):
    print (bvh_data.shape)
    lowest_points = []
    hip_index = joint_index['hip']
    left_foot_index = joint_index['lFoot']
    left_nub_index = joint_index['lFoot_Nub']
    right_foot_index = joint_index['rFoot']
    right_nub_index = joint_index['rFoot_Nub']
                
                
    for i in range(bvh_data.shape[0]):
        frame = bvh_data[i,:]
        #print 'hi1'
        foot_heights = [frame[left_foot_index*3+1],frame[left_nub_index*3+1],frame[right_foot_index*3+1],frame[right_nub_index*3+1]]
        lowest_point = min(foot_heights) + frame[hip_index*3 + 1]
        lowest_points.append(lowest_point)
        
                                
        #print lowest_point
    lowest_points = sort(lowest_points)
    num_frames = bvh_data.shape[0]
    quarter_length = int(num_frames/4)
    end = 3*quarter_length
    overall_lowest = mean(lowest_points[quarter_length:end])
    
    return overall_lowest

def sanity():
    for i in range(4):
        print ('hi')
        
 
def get_motion_center(bvh_data):
    center=np.zeros(3)
    for frame in bvh_data:
        center=center+frame[0:3]
    center=center/bvh_data.shape[0]
    return center

def augment_positional_frame(train_frame_data, T, axisR):
    # Input series of hip positions without the hip [, 171] for a single motion frame
    hip_index=joint_index['hip']
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3) ):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]+hip_pos 
    
    
    mat_r_augment=euler.axangle2mat(axisR[0:3], axisR[3])
    n=int(len(train_frame_data)/3)
    for i in range(n): # for each frame
        raw_data=train_frame_data[i*3:i*3+3]
        new_data = np.dot(mat_r_augment, raw_data)+T
        train_frame_data[i*3:i*3+3]=new_data
    
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3)):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]-hip_pos
    
    return train_frame_data
   
def augment_positional(train_data, T, axisR):
    result=list(map(lambda frame: augment_positional_frame(frame, T, axisR), train_data))
    return np.array(result) 

#########################################################################
#########################################################################
## Skeleton to representation ##########################################
#########################################################################
#########################################################################  
  
#input a vector of data, with the first three data as translation and the rest the euler rotation
#output a vector of data, with the first three data as translation not changed and the rest to quaternions.
#note: the input data are in z, x, y sequence
def load_frame_as_positional(raw_frame_data, non_end_bones, skeleton):
    pos_dic=  helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data= np.zeros(len(pos_dic.keys())*3)
    i=0
    hip_pos=pos_dic['hip']
    #print hip_pos

    for joint in pos_dic.keys():
        if(joint=='hip'):
            
            new_data[i*3:i*3+3]=pos_dic[joint].reshape(3)
        else:
            new_data[i*3:i*3+3]=pos_dic[joint].reshape(3)- hip_pos.reshape(3)
        i=i+1
    #print new_data
    new_data=new_data*0.01
    return new_data

def load_frame_as_eulers(raw_frame_data, non_end_bones, skeleton, augmented_pos):
    # Extraction of the hip position
    augmented_hip = augmented_pos[:3]

    # Extraction of the rotations
    rotation_dict = helper.get_skeleton_rotations(raw_frame_data, non_end_bones, skeleton)

    new_data = np.zeros(len(rotational_joints_index)*3 + 3)
    for joint in rotational_joints_index.keys():
        if rotation_dict[joint] is None: continue # Teminal bones have no rotations
        i = rotational_joints_index[joint]
        new_data[3+i*3:i*3+3+3]=np.fromiter(rotation_dict[joint], dtype=np.float) * (math.pi / 180) # from grades to radiants

    new_data[:3] = augmented_hip # append hip at head of the vector where the model expects it

    return new_data
    
def load_frame_as_6d(raw_frame_data, non_end_bones, skeleton, augmented_pos):
    # Extraction of the hip position
    augmented_hip = augmented_pos[:3]

    # Extraction of the rotations
    rotation_dict = helper.get_skeleton_rotations(raw_frame_data, non_end_bones, skeleton)

    new_data = np.zeros(len(rotational_joints_index)*6 + 3)
    for joint in rotational_joints_index.keys():
        if rotation_dict[joint] is None: continue # Teminal bones have no rotations
        i = rotational_joints_index[joint]
        euler_rotation = torch.tensor(rotation_dict[joint], dtype=float) * (math.pi / 180) #load angles are in zxy  + from grades to radiants
        rot_matrix = rot_conv.euler_angles_to_matrix(euler_rotation,"ZXY")
        d6_rep = rot_conv.matrix_to_rotation_6d(rot_matrix).numpy()
        new_data[3+i*6:i*6+6+3] = d6_rep

    new_data[:3] = augmented_hip # append hip at head of the vector where the model expects it

    return new_data

def load_frame_as_quaternions(raw_frame_data, non_end_bones, skeleton, augmented_pos):
    # Extraction of the hip position
    augmented_hip = augmented_pos[:3]

    # Extraction of the rotations
    rotation_dict = helper.get_skeleton_rotations(raw_frame_data, non_end_bones, skeleton)

    new_data = np.zeros(len(rotational_joints_index)*4 + 3)
    for joint in rotational_joints_index.keys():
        if rotation_dict[joint] is None: continue # Teminal bones have no rotations
        i = rotational_joints_index[joint]
        euler_rotation = torch.tensor(rotation_dict[joint], dtype=float) * (math.pi / 180) #load angles are in zxy  + from grades to radiants
        rot_matrix = rot_conv.euler_angles_to_matrix(euler_rotation, "ZXY")
        quaternions = rot_conv.matrix_to_quaternion(rot_matrix).numpy()
        new_data[3+i*4:i*4+4+3] = quaternions

    new_data[:3] = augmented_hip # append hip at head of the vector where the model expects it

    return new_data
 
def get_training_format_data(raw_data, non_end_bones, skeleton, loader):
    new_data=[]
    for frame in raw_data:
        new_frame=loader(frame, non_end_bones, skeleton)
        new_data=new_data+[new_frame]
    return np.array(new_data)



def get_weight_dict(skeleton):
    weight_dict=[]
    for joint in skeleton:
        parent_number=0.0
        j=joint
        while (skeleton[joint]['parent']!=None):
            parent_number=parent_number+1
            joint=skeleton[joint]['parent']
        weight= pow(math.e, -parent_number/5.0)
        weight_dict=weight_dict+[(j, weight)]
    return weight_dict

def get_rotation_weight_dict():
    weight_dict=[]
    for joint in rotational_joints_index:
        parent_number=0.0
        while (skeleton[joint]['parent']!=None):
            parent_number=parent_number+1
            joint=skeleton[joint]['parent']
        weight = parent_number+1
        weight_dict=weight_dict+[weight]

    return weight_dict

 
def get_training_format_data_rotational(raw_data, non_end_bones, skeleton, augmented_pos, loader):
    new_data=[]
    for frame_id, frame in enumerate(raw_data):
        new_frame=loader(frame, non_end_bones, skeleton, augmented_pos[frame_id])
        new_data=new_data+[new_frame]
    return np.array(new_data)

### DATA PROCESSING ENTRY POINT #################
def get_train_data(bvh_filename, representation):
    data=parse_frames(bvh_filename)
    train_data=get_training_format_data(data, non_end_bones,skeleton, load_frame_as_positional) #Extract for each frame the list of joint position compressed in 1-dim array shape (171)

    center=get_motion_center(train_data) #get the avg position of the hip
    center[1]=0.0 #don't center the height
    train_data = augment_positional(train_data, -center, [0,1,0, 0.0]) # Position have to be normalized
    
    # We need the processd positional data in order to have the hip translated pos for rotational data
    if representation == 'positional':
        train_data = train_data
    elif representation == 'euler':
        train_data = get_training_format_data_rotational(data, non_end_bones,skeleton, train_data, load_frame_as_eulers)
    elif representation == '6d':
        train_data = get_training_format_data_rotational(data, non_end_bones,skeleton, train_data, load_frame_as_6d)
    elif representation == 'quaternions':
        train_data = get_training_format_data_rotational(data, non_end_bones,skeleton, train_data, load_frame_as_quaternions)
    else:
        raise Exception(f"Invalid data representation:{representation} -> allowed [positional, euler, 6d, quaternions]")
    

    return train_data

          

def write_frames(format_filename, out_filename, data):
    
    format_lines = get_frame_format_string(format_filename)
    
    num_frames = data.shape[0]
    format_lines[len(format_lines)-2]="Frames:\t"+str(num_frames)+"\n"
    
    bvh_file = open(out_filename, "w")
    bvh_file.writelines(format_lines)
    bvh_data_str=vectors2string(data)
    bvh_file.write(bvh_data_str)    
    bvh_file.close()

def regularize_angle(a):
	
	if abs(a) > 180:
		remainder = a%180
		print ('hi')
	else: 
		return a
	
	new_ang = -(sign(a)*180 - remainder)
	
	return new_ang

def write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, output_filename):
    bvh_vec_length = len(non_end_bones)*3 + 6
    
    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debug(skeleton, positions)
        new_motion1 = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, positions)
								
        new_motion = np.array([round(a,6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]
								
        out_data[i,:] = np.transpose(new_motion[:,np.newaxis])
        
    
    write_frames(format_filename, output_filename, out_data)

def write_rotation_to_bvh(rotation_angles, hip_pos, non_end_bones, format_filename, output_filename):
    bvh_vec_length = len(non_end_bones)*3 + 6
    
    out_data = np.zeros([len(rotation_angles), bvh_vec_length])
    for i in range(1, len(rotation_angles)):
        frame_rotations = rotation_angles[i]
        hip = hip_pos[i]

        new_motion1 = helper.rotation_dic_to_vec_hip(frame_rotations, non_end_bones, hip)
								
        new_motion = np.array([round(a,6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]
								
        out_data[i,:] = np.transpose(new_motion[:,np.newaxis])
        
    
    write_frames(format_filename, output_filename, out_data)


# Write switcher based on datatype
def write_traindata_to_bvh(bvh_filename, train_data, method):
    if method == 'positional':
        write_position_to_bvh(bvh_filename, train_data)
    elif method == 'euler':
        write_eulers_to_bvh(bvh_filename, train_data)
    elif method == '6d':
        write_6d_to_bvh(bvh_filename, train_data)
    elif method == 'quaternions':
        write_quaternions_to_bvh(bvh_filename, train_data)
    else:
        raise Exception(f"Invalid data representation:{method} -> allowed [positional, euler, 6d, quaternions]")

## POSITIONAL DATA save
def write_position_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    xyz_motion = []
    format_filename = standard_bvh_file
    for i in range(seq_length):
        data = train_data[i]
        data = np.array([round(a,6) for a in train_data[i]])
        #print data
        #input(' ' )
        position = data_vec_to_position_dic(data, skeleton)        
        
        
        xyz_motion.append(position)

        
    write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename)

def data_vec_to_position_dic(data, skeleton):
    data = data*100
    hip_pos=data[joint_index['hip']*3:joint_index['hip']*3+3]
    positions={}
    for joint in joint_index:
        positions[joint]=data[joint_index[joint]*3:joint_index[joint]*3+3]
    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
            
    return positions

 ## POSITIONAL DATA save - END


############################################################
## Euler Data save
##########################################################3
def write_eulers_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    
    format_filename = standard_bvh_file
    sequence_rotations = []
    sequence_hip_pos = []
    for i in range(seq_length):
        data = train_data[i]
        data = np.array([round(a,6) for a in train_data[i]])
        rotation, hip = euler_data_extraction(data)        
        sequence_rotations.append(rotation)
        sequence_hip_pos.append(hip)
 
    write_rotation_to_bvh(sequence_rotations, sequence_hip_pos, non_end_bones, format_filename, bvh_filename)

def euler_data_extraction(data):
    hip_pos=data[:3]*100 # Get position of the hip
    
    rotations={}
    for joint in rotational_joints_index.keys():
        idx = rotational_joints_index[joint]
        rotations[joint]=data[3+idx*3:idx*3+3+3] * (180.0 / math.pi)
    
    return rotations, hip_pos
## Euler Data save end


############################################################
## 6D Data save
##########################################################3
def write_6d_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    
    format_filename = standard_bvh_file
    sequence_rotations = []
    sequence_hip_pos = []
    for i in range(seq_length):
        data = train_data[i]
        data = np.array([round(a,6) for a in train_data[i]])
        rotation, hip = d6_data_extraction(data)        
        sequence_rotations.append(rotation)
        sequence_hip_pos.append(hip)
 
    write_rotation_to_bvh(sequence_rotations, sequence_hip_pos, non_end_bones, format_filename, bvh_filename)

def d6_data_extraction(data):
    hip_pos=data[:3]*100 # Get position of the hip
    
    rotations={}
    for joint in rotational_joints_index.keys():
        idx = rotational_joints_index[joint]
        d6_rotation = torch.from_numpy(data[3+idx*6:idx*6+6+3])
        rot_mat = rot_conv.rotation_6d_to_matrix(d6_rotation)
        euler_angles = rot_conv.matrix_to_euler_angles(rot_mat, 'ZXY').numpy() * (180.0 / math.pi)
        rotations[joint]=euler_angles
    
    return rotations, hip_pos

############################################################
## Quaternions Data save
##########################################################3
def write_quaternions_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    
    format_filename = standard_bvh_file
    sequence_rotations = []
    sequence_hip_pos = []
    for i in range(seq_length):
        data = train_data[i]
        data = np.array([round(a,6) for a in train_data[i]])
        rotation, hip = quaternions_data_extraction(data)        
        sequence_rotations.append(rotation)
        sequence_hip_pos.append(hip)
 
    write_rotation_to_bvh(sequence_rotations, sequence_hip_pos, non_end_bones, format_filename, bvh_filename)

def quaternions_data_extraction(data):
    hip_pos=data[:3]*100 # Get position of the hip
    
    rotations={}
    for joint in rotational_joints_index.keys():
        idx = rotational_joints_index[joint]
        quaternion = torch.from_numpy(data[3+idx*4:idx*4+4+3])
        rot_mat = rot_conv.quaternion_to_matrix(quaternion)
        euler_angles = rot_conv.matrix_to_euler_angles(rot_mat, 'ZXY').numpy() * (180.0 / math.pi)
        rotations[joint]=euler_angles
    
    return rotations, hip_pos


###################################################################

# Forward Kinematics
def fk(hip_pos, rotation_matrices):
    # prepare motion data
    motion = np.zeros((132))
    motion[:3] = hip_pos * 100
    for joint in rotational_joints_index.keys():
        idx = rotational_joints_index[joint]
        motion[3+idx*3: 3 + idx*3 + 3] = rot_conv.matrix_to_euler_angles(rotation_matrices[idx], 'ZXY') * (180.0 / math.pi)

    fk_pos = helper.get_skeleton_position(motion, non_end_bones, skeleton)

    pos_array = np.zeros((171))
    idx = 0
    for joint in fk_pos.keys():
        pos_array[idx*3: idx*3 + 3] = fk_pos[joint].flatten()
        idx+=1

    return pos_array * 0.01
       
def get_pos_dic(frame, joint_index):
    positions={}
    for key in joint_index.keys():
        positions[key]=frame[joint_index[key]*3:joint_index[key]*3+3]
    return positions


def vector2string(data):
    s=' '.join(map(str, data))
    
    return s

def vectors2string(data):
    s='\n'.join(map(vector2string, data))
   
    return s
 
    
def get_child_list(skeleton,joint):
    child=[]
    for j in skeleton:
        parent=skeleton[j]['parent']
        if(parent==joint):
            child.append(j)
    return child
    
def get_norm(v):
    return np.sqrt( v[0]*v[0]+v[1]*v[1]+v[2]*v[2] )

def get_regularized_positions(positions):
    org_positions=positions
    new_positions=regularize_bones(org_positions, positions, skeleton, 'hip')
    return new_positions

def regularize_bones(original_positions, new_positions, skeleton, joint):
    children=get_child_list(skeleton, joint)
    for child in children:
        offsets=skeleton[child]['offsets']
        length=get_norm(offsets)
        direction=original_positions[child]-original_positions[joint]
        #print child
        new_vector=direction*length/get_norm(direction)
        #print child
        #print length, get_norm(direction)
        #print new_positions[child]
        new_positions[child]=new_positions[joint]+new_vector
        #print new_positions[child]
        new_positions=regularize_bones(original_positions,new_positions,skeleton,child)
    return new_positions

def get_regularized_train_data(one_frame_train_data):
    
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
            
    
    new_pos=get_regularized_positions(positions)
    
    
    new_data=np.zeros(one_frame_train_data.shape)
    i=0
    for joint in new_pos.keys():
        if (joint!='hip'):
            new_data[i*3:i*3+3]=new_pos[joint]-new_pos['hip']
        else:
            new_data[i*3:i*3+3]=new_pos[joint]
        i=i+1
    new_data=new_data*0.01
    return new_data

def check_length(one_frame_train_data):
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
    
    for joint in positions.keys():
        if(skeleton[joint]['parent']!=None):
            p1=positions[joint]
            p2=positions[skeleton[joint]['parent']]
            b=p2-p1
            #print get_norm(b), get_norm(skeleton[joint]['offsets'])
    


standard_bvh_file="train_data_bvh/standard.bvh"
weight_translation=0.01
skeleton, non_end_bones=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)    
sample_data=parse_frames(standard_bvh_file)
joint_index= get_pos_joints_index(sample_data[0],non_end_bones, skeleton)
rotational_joints_index = dict(zip(['hip'] + non_end_bones, range(len(non_end_bones)+1)))