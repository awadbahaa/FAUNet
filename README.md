# FAUNet
Official Implementaion of FAUNet
This Github repo is still a work in progress.
However, FAUNet_Train can be used to train the network. 
To run the code, use the following command, just make sure to setup the paths and dataset correctly.
```bash
python FAUNet_Tain.py
```

while FAUNet_Test can be used to conduct inference on a folder containing test images. 
```bash
python FAUNet_Test.py
```
# FAUNet
Note: the code will work on an Nvidia gpu currently. since the high pass filter has to be sent to the device selected at the moment the netwrok is created. 

# Data structure and preparation 
The strucutre of the data folder is the same as seen in data_set. 
The code [here](https://github.com/chrieke/InstanceSegmentation_Sentinel2) can be used to prepare the data in this format. 

## Acknowledgments
A big thank you to [milesial](https://github.com/milesial) for the original [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) implementation. Any modifications made in this fork are built upon the solid foundation they provided.

