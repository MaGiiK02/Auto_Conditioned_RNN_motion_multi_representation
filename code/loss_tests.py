import torch
import utils.losses as losses
import math
import utils.rotations_conversion as conv

ROTATION_1 = [1,0,0]
ROTATION_2 = [1,0,0]
ROTATION_3 = [0,1,0]

ROTATION_1_TARGET = [1,0,0]
ROTATION_2_TARGET = [0,0,1]
ROTATION_3_TARGET = [0,0,0]

## Uncomment for eqaul test
ROTATION_1_TARGET = [1,0,0]
ROTATION_2_TARGET = [1,0,0]
ROTATION_3_TARGET = [0,1,0]

if __name__ == '__main__':
    rotation_src = torch.tensor([
        [ROTATION_1, ROTATION_2, ROTATION_3], 
        [ROTATION_3, ROTATION_2, ROTATION_1],
    ], dtype=float) * (180.0 / math.pi)

    rotation_target = torch.tensor([
        [ROTATION_1_TARGET, ROTATION_2_TARGET, ROTATION_3_TARGET], 
        [ROTATION_3_TARGET, ROTATION_2_TARGET, ROTATION_1_TARGET]
    ], dtype=float) * (180.0 / math.pi)

    rotation_src_mat = conv.euler_angles_to_matrix(rotation_src, 'ZYX')
    rotation_target_mat = conv.euler_angles_to_matrix(rotation_target, 'ZYX')

    ## EULER
    rotation_src = rotation_src.reshape((2,-1))
    rotation_target = rotation_target.reshape((2,-1))
    print(losses.angle_distance(rotation_src, rotation_target))

    ## 6D
    d6_src = conv.matrix_to_rotation_6d(rotation_src_mat).reshape((2,-1))
    d6_target = conv.matrix_to_rotation_6d(rotation_target_mat).reshape((2,-1))
    print(losses.mae(d6_src, d6_target))

    ## QUATERNIONS
    q_src = conv.matrix_to_quaternion(rotation_src_mat).reshape((2,-1))
    q_target = conv.matrix_to_quaternion(rotation_target_mat).reshape((2,-1))
    print(losses.quaternion_loss(q_src, q_target))