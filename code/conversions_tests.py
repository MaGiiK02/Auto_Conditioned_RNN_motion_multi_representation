import read_bvh
import shutil
import os

INPUT = '/home/mangelini/Develop/UCY/Auto_Conditioned_RNN_motion/train_data_bvh/salsa/01.bvh'
OUT = '/home/mangelini/Develop/UCY/Auto_Conditioned_RNN_motion/tests/conversion'
REPS = ['positional', 'euler', '6d', 'quaternions',]

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(INPUT, f'{OUT}/original.bvh')
    for r in REPS:
        loaded_bvh = read_bvh.get_train_data(INPUT, r)
        read_bvh.write_traindata_to_bvh(f'{OUT}/test_{r}.bvh',loaded_bvh, r)
