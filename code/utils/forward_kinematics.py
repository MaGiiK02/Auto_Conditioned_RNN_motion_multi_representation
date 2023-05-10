import torch
import utils.rotations_conversion as rot_conv
from read_bvh import fk

def fk_euler(sequence_batch: torch.Tensor, sequence_length):
    frame_length = 3 + 57*3
    fk_pos = torch.zeros((sequence_batch.shape[0], sequence_length, 171))
    for b in range(sequence_batch.shape[0]):
        sequence = sequence_batch[b].reshape((sequence_length, frame_length))
        for f in range(sequence_length):
            frame = sequence[f]
            pos_hip = frame[:3]
            rotations = frame[3:]
            mat_rot = []
            for r in range(int((len(rotations)/3))):
                mat_rot.append(rot_conv.euler_angles_to_matrix(rotations[r*3: r*3+3], 'ZXY'))
            
            fk_pos[b][f] = torch.Tensor(fk(pos_hip, mat_rot))

    return fk_pos

def fk_6D(sequence_batch: torch.Tensor, sequence_length):
    frame_length = 3 + 57*6
    fk_pos = torch.zeros((sequence_batch.shape[0], sequence_length, 171))
    for b in range(sequence_batch.shape[0]):
        sequence = sequence_batch[b].reshape((sequence_length, frame_length))
        for f in range(sequence_length):
            frame = sequence[f]
            pos_hip = frame[:3]
            rotations = frame[3:]
            mat_rot = []
            for r in range(int((len(rotations)/6))):
                mat_rot.append(rot_conv.rotation_6d_to_matrix(rotations[r*6: r*6+6]))
            
            fk_pos[b][f] = torch.Tensor(fk(pos_hip, mat_rot))

    return fk_pos

def fk_quaternions(sequence_batch: torch.Tensor, sequence_length):
    frame_length = 3 + 57*4
    fk_pos = torch.zeros((sequence_batch.shape[0], sequence_length, 171))
    for b in range(sequence_batch.shape[0]):
        sequence = sequence_batch[b].reshape((sequence_length, frame_length))
        for f in range(sequence_length):
            frame = sequence[f]
            pos_hip = frame[:3]
            rotations = frame[3:]
            mat_rot = []
            for r in range(int((len(rotations)/4))):
               mat_rot.append(rot_conv.quaternion_to_matrix(rotations[r*4: r*4+4]))
            
            fk_pos[b][f] = torch.Tensor(fk(pos_hip, mat_rot))

    return fk_pos