# acLSTM_motion
This folder contains an implementation of acRNN tha is also adapted to use different representations as Euler, 6D and Quaternions over the Positional representation if motion the origianl paper use.

[Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis](https://arxiv.org/abs/1707.05363)

[CMU Motion Capture Database](http://mocap.cs.cmu.edu/)

You can find here some pretrained models, and some visualization of the results: [Drive Folder](https://drive.google.com/drive/folders/1JfhW0OaYJZGgJEH7siPrV4vnqBNskqpi?usp=share_link)


### Prequisite

You can find the neccesary libraries in the `enviorment.yml` and `requirments.txt` files:
If you use conda:
```bash
conda env create -f environment.yml
```
if you have python 3.6 
```bash
pip install -r requirements.txt
```

### VS-CODE (Racomended)
This project has a reaady to use *lauch.json* config for VS-CODE, as such you will have all the configurations ready to run on the debug page.

### Data Preparation

To begin, you need to download the motion data form the CMU motion database in the form of bvh files. I have already put some sample bvh files including "salsa", "martial" and "indian" in the "train_data_bvh" folder. Or use the bvh file already present inside the `./train_data_bvh` folder.

Then to transform the bvh files into training data, go to the folder "code" and run [generate_training_data.py](code/generate_training_data.py), it will convert the data and keep 30% of sequencies for testing.

Example:
```bash 
python3 train_data_bvh/martial/ train_data_xyz/positional --representation=positional
```
Accpeted representatins are (case sensitive):
* positional
* euler
* 6d
* quaternions

**NOTE:** Be sure to run the different script from th parent folder of `./code/` or the program will crash.

### Training

After generating the training data, you can start to train the network by running the [pytorch_train_aclstm.py](code/pytorch_train_aclstm.py). 

Example:
```bash 
python3 ./code/pytorch_train_aclstm.py --representation=positional --dances_folder=train_data_xyz/positional/ --write_weight_folder=run/postitional/weigths/ --write_bvh_motion_folder=runs/positional/bvh/ --dance_frame_rate=120 --batch_size=32 --epochs=50000--representation=positional
```
```--read_weight_path=``` can be used to resume train from a checkpoint.

### Eval
When the training is done, you can use [pytorch_test_prediction.py](code/pytorch_test_prediction.py) to evaluate the model prediction on x frame. 

Example:
```bash 
python3 ./code/pytorch_test_prediction.py   --dances_folder=train_data_xyz/positional/test/ --read_weight_path=runs/positional/weigths/0025000.weight --out_folder=eval/positional/ --dance_frame_rate=120 --dances_test_size=10 --representation=positional
```

Note: to keep x low or the script wil crash by beign uable to aquire enough reference frames.

### Syntesis

When the training is done, you can use [pytorch_test_synthesize_motion.py](code/pytorch_test_synthesize_motion.py) to synthesize motions.

Example:
```bash 
python3 ./code/pytorch_test_synthesize_motion.py --dances_folder=train_data_xyz/positional/test/ --read_weight_path=runs/positional/weigths/0025000.weight --write_bvh_motion_folder=eval/positional/bvh/ --dance_frame_rate=120 --dances_test_size=10 -representation=positional
```

The output motions from the network usually have artifacts of sliding feet and sometimes underneath-ground feet. If you are not satisfied with these details, you can use [fix_feet.py](code/fix_feet.py) to solve it. The algorithm in this code is very simple and you are welcome to write a more complex version that can preserve the kinematics of the human body and share it to us.

For rendering the bvh motion, you can use softwares like MotionBuilder, Maya, 3D max or most easily, use an online BVH renderer for example:
http://lo-th.github.io/olympe/BVH_player.html 



Enjoy!
