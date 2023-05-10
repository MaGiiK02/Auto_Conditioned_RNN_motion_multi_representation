import torch

# split positional hp data from rotational to apply different losses
def getHipAndRotationsFromSequence(batch_sequence, joint_size, frame_hip_index, sequence_length):
    positions = torch.zeros((batch_sequence.shape[0], sequence_length*3))
    rotations = torch.zeros((batch_sequence.shape[0], batch_sequence.shape[1] - sequence_length*3))
    for b in range(batch_sequence.shape[0]):
        for s in range(sequence_length):
            sequence_shift = (joint_size*57 + 3) * s
            sequence_shift_pos = 3 * s
            sequence_shift_rot = (joint_size*57) * s
            positions[b][sequence_shift_pos:sequence_shift_pos+3] = batch_sequence[b,sequence_shift + frame_hip_index*joint_size: sequence_shift + frame_hip_index*joint_size +3]
            rotations[b][sequence_shift_rot:sequence_shift_rot+(57*joint_size)] = torch.cat([
                batch_sequence[b,sequence_shift:sequence_shift + frame_hip_index*joint_size],
                batch_sequence[b,sequence_shift + frame_hip_index*joint_size +3:3+sequence_shift+((57-frame_hip_index)*joint_size)]
            ], dim=0)
        
    return rotations, positions