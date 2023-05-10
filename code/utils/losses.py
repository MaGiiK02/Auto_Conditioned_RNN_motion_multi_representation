import torch
import math
import torch.nn.functional as tf
from utils.rotations_conversion import standardize_quaternion 

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
def joint_weighted_angle_distance(x: torch.Tensor, y: torch.Tensor, joint_weight_map):
    error = 1 - torch.cos(x - y)
    error = x.reshape((x.shape[0], -1, 6))
    weighted_error = torch.einsum('bje,i->bje', error, joint_weight_map).reshape(x.shape[0],-1)
    return torch.mean(weighted_error)

def joint_weighted_mae_6d(x: torch.Tensor, y: torch.Tensor, joint_weight_map):
    error = torch.abs(x - y)
    error = x.reshape((x.shape[0], -1, 6))
    weighted_error = torch.einsum('bje,i->bje', error, joint_weight_map).reshape(x.shape[0],-1)
    return torch.mean(weighted_error)

def joint_weighted_quaternion_loss(x: torch.Tensor, y: torch.Tensor, joint_weight_map):
    x = x.reshape((x.shape[0], -1, 4))
    y = y.reshape((y.shape[0], -1, 4))

    x = torch.nn.functional.normalize(x, p=2.0, dim = 2) # quaternion normalization

    angle = torch.mul(x,y).sum(dim=2).reshape(x.shape[0],-1) # quaternion wise dot product

    #fix to avoid derivate explosion
    eps = 1e-7
    angle = angle.clamp(-1+ eps, 1-eps)

    # Calc angle error
    error = 2 * torch.arccos(angle.abs())

    weighted_error = torch.einsum('bje,i->bje', error, joint_weight_map).reshape(x.shape[0],-1)
    return torch.mean(weighted_error) 