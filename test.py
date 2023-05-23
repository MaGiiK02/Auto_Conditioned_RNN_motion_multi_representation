import glob
import cv2
import os
import numpy as np

CROP_SIZE = 512

CROP_X_START = 940-int(CROP_SIZE/2)
CROP_X_END = CROP_X_START + CROP_SIZE
CROP_Y_START = 540-int(CROP_SIZE/2)
CROP_Y_END = CROP_Y_START + CROP_SIZE

METHODS = [
    '6DW',
    'EulerW',
    'quatw',
    '6D',
    'Pos',
    'Euler',
    'quat',
    'Start'
]

METHODS_DICT = {
    '6DW':5,
    'EulerW':4,
    'quatw':6,
    '6D': 2,
    'Pos': 0,
    'Euler': 1,
    'quat': 3,
    'Start': -1
}

METHODS_COUNT= {
    '6DW':1,
    'EulerW':1,
    'quatw':1,
    '6D': 1,
    'Pos': 1,
    'Euler': 1,
    'quat': 1,
    'Start': -1
}


def get_method(path):
    for m in METHODS:
        if m in path: return m
    raise Exception("Nomethod found!")

if __name__ == '__main__' :
    mosaic = np.zeros((7*CROP_SIZE, 5*CROP_SIZE, 3))
    images_paths = glob.glob("/home/mangelini/Desktop/Images Qualitative/*.png", recursive=True)
    images_paths.sort()
    for image_path in images_paths:
        image = cv2.imread(image_path)
        cropped = image[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

        folder = os.path.dirname(image_path)
        name = os.path.basename(image_path)
        m = get_method(image_path)
        out_path = f'{folder}/{m}/'
        os.makedirs(out_path, exist_ok=True)
        cv2.imwrite(f"{out_path}{name}", cropped)

        if (m == 'Start'):
            for i in range(7):
                mosaic[i*CROP_SIZE:i*CROP_SIZE+CROP_SIZE, 0:CROP_SIZE] = cropped
        
        elif ('quat' in m ):
            m_index = METHODS_DICT[m]
            for i in range(1, 5):
                mosaic[m_index*CROP_SIZE:m_index*CROP_SIZE+CROP_SIZE, i*CROP_SIZE:i*CROP_SIZE+CROP_SIZE] = cropped
        else:
            m_index = METHODS_DICT[m]
            anim_id = METHODS_COUNT[m]
            mosaic[m_index*CROP_SIZE:m_index*CROP_SIZE+CROP_SIZE, anim_id*CROP_SIZE:anim_id*CROP_SIZE+CROP_SIZE] = cropped
            METHODS_COUNT[m] = METHODS_COUNT[m] +1

    cv2.imwrite("/home/mangelini/Desktop/Images Qualitative/mosaic.png", mosaic)
