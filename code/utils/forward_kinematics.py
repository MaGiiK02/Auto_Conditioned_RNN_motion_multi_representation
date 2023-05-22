import torch
import utils.rotations_conversion as rot_conv
import read_bvh
import numpy as np

rot_joint = len(read_bvh.rotational_joints_index)

def fk_euler(sequence_batch: torch.Tensor):
    frame_size = 3 + rot_joint*3
    out = []
    for b in range(sequence_batch.shape[0]):
        out_seq=np.array(sequence_batch[b].data.tolist()).reshape(-1,frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,0*3]=out_seq[frame,0*3]+last_x
            last_x=out_seq[frame,0*3]
            
            out_seq[frame,0*3+2]=out_seq[frame,0*3+2]+last_z
            last_z=out_seq[frame,0*3+2]

        read_bvh.write_traindata_to_bvh("./temp_conversion.bvh", out_seq, "euler")
        out.append(read_bvh.get_train_data("./temp_conversion.bvh", "positional"))
    return out


def fk_6D(sequence_batch: torch.Tensor):
    frame_size = 3 + rot_joint*6
    out = []
    for b in range(sequence_batch.shape[0]):
        out_seq=np.array(sequence_batch[b].data.tolist()).reshape(-1,frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,0*3]=out_seq[frame,0*3]+last_x
            last_x=out_seq[frame,0*3]
            
            out_seq[frame,0*3+2]=out_seq[frame,0*3+2]+last_z
            last_z=out_seq[frame,0*3+2]

        read_bvh.write_traindata_to_bvh("./temp_conversion.bvh", out_seq, "6d")
        out.append(read_bvh.get_train_data("./temp_conversion.bvh", "positional"))
    return out

def fk_quaternions(sequence_batch: torch.Tensor):
    frame_size = 3 + rot_joint*4
    out = []
    for b in range(sequence_batch.shape[0]):
        out_seq=np.array(sequence_batch[b].data.tolist()).reshape(-1,frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,0*3]=out_seq[frame,0*3]+last_x
            last_x=out_seq[frame,0*3]
            
            out_seq[frame,0*3+2]=out_seq[frame,0*3+2]+last_z
            last_z=out_seq[frame,0*3+2]

        read_bvh.write_traindata_to_bvh("./temp_conversion.bvh", out_seq, "quaternions")
        out.append(read_bvh.get_train_data("./temp_conversion.bvh", "positional"))
    return out