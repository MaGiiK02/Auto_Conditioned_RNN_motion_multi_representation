import torch
import torch.nn.functional as tf
from read_bvh import get_rotation_weight_dict, rotational_joints_index

Rotational_joint = len(rotational_joints_index)
LOSS_WEIGHTS = torch.Tensor(get_rotation_weight_dict())

def angle_distance(x: torch.Tensor, y: torch.Tensor):
    cos_error = torch.cos(x - y)
    return torch.mean(1 - cos_error)

def mae(x: torch.Tensor, y: torch.Tensor):
    return tf.l1_loss(x, y)

def quaternion_loss(x: torch.Tensor, y: torch.Tensor):
    x = x.reshape((x.shape[0], -1, 4))
    y = y.reshape((y.shape[0], -1, 4))

    x = torch.nn.functional.normalize(x, p=2.0, dim = 2) # quaternion normalization

    angle = torch.mul(x,y).sum(dim=2).reshape(x.shape[0],-1) # quaternion wise dot product

    #fix to avoid derivate explosion
    eps = 1e-7
    angle = angle.clamp(-1+ eps, 1-eps)

    return torch.mean(2 * torch.arccos(angle.abs())) 


# Personal Experiment
def joint_weighted_angle_distance(x: torch.Tensor, y: torch.Tensor):
    error = 1 - torch.cos(x - y)
    error = error.reshape((error.shape[0], -1, Rotational_joint, 3))
    single_angle_error = error.mean(3) # Mean the error along the angles of the joint to apply the weighted loss
    weighted_error = torch.einsum('bsj,j->bsj', single_angle_error, 1/LOSS_WEIGHTS).reshape(x.shape[0],-1)
    return torch.mean(weighted_error)

def joint_weighted_mae_6d(x: torch.Tensor, y: torch.Tensor):
    error = torch.abs(x - y)
    error = error.reshape((error.shape[0], -1, Rotational_joint, 6))
    single_angle_error = error.mean(3) # Mean the error along the angles of the joint to apply the weighted loss
    weighted_error = torch.einsum('bsj,j->bsj', single_angle_error, 1/LOSS_WEIGHTS).reshape(x.shape[0],-1)
    return torch.mean(weighted_error)

def joint_weighted_quaternion_loss(x: torch.Tensor, y: torch.Tensor):
    x = x.reshape((x.shape[0], -1, 4))
    y = y.reshape((y.shape[0], -1, 4))

    x = torch.nn.functional.normalize(x, p=2.0, dim = 2) # quaternion normalization

    angle = torch.mul(x,y).sum(dim=2).reshape(x.shape[0],-1) # quaternion wise dot product

    #fix to avoid derivate explosion
    eps = 1e-7
    angle = angle.clamp(-1+ eps, 1-eps)

    # Calc angle error
    error = 2 * torch.arccos(angle.abs())
    error = error.reshape(((x.shape[0], -1, Rotational_joint)))
    weighted_error = torch.einsum('bsj,j->bsj', error, 1/LOSS_WEIGHTS).reshape(x.shape[0],-1)
    return torch.mean(weighted_error) 