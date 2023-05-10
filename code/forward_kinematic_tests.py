import read_bvh
import shutil
import os
import torch
import utils.forward_kinematics as fk

INPUT = '/home/mangelini/Develop/UCY/Auto_Conditioned_RNN_motion/train_data_bvh/salsa/01.bvh'
OUT= '/home/mangelini/Develop/UCY/Auto_Conditioned_RNN_motion/tests/forward_kinematics/'
REPS = ['euler', '6d', 'quaternions', ]

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(INPUT, f'{OUT}/original.bvh')
    target = torch.Tensor([read_bvh.get_train_data(INPUT, 'positional')])
    read_bvh.write_traindata_to_bvh(f'{OUT}/rf_kinemeatic.bvh', target[0].numpy(), 'positional')
    for r in REPS:
        loaded_bvh = torch.Tensor([read_bvh.get_train_data(INPUT, r)])

        frames = loaded_bvh.shape[1]
        loaded_bvh = loaded_bvh.reshape((1,-1))
        fk_predicted = None
        if('euler' == r): fk_predicted = fk.fk_euler(loaded_bvh, frames)
        elif('6d' == r): fk_predicted = fk.fk_6D(loaded_bvh, frames)
        elif('quaternions' == r): fk_predicted = fk.fk_quaternions(loaded_bvh, frames)

        print("MSE Positions:", torch.mean(torch.sqrt(torch.square(target.reshape((1,-1))-fk_predicted.reshape((1,-1))))).item())
        read_bvh.write_traindata_to_bvh(f'{OUT}/test_{r}.bvh', fk_predicted[0].numpy(), 'positional')