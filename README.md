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
# Notes
1.  The trained network will only work on the device type it was originally trained on, which is currently an Nvidia GPU. This is because the high pass filter weights must be sent to the selected device during network initialization. If you train it on another device type (such as CPU or MPS), the inference has to be conducted on that device.

2.  The dataset is too big to be uploaded here. I will make it available on a cloud storage platform soon. If I haven't uploaded it yet and you need the data, just email me, and I will speed up the process.

# Data structure and preparation 
The structure of the dataset should be the same as seen in `data_set`. The code [here](https://github.com/chrieke/InstanceSegmentation_Sentinel2) can be used to prepare the data in this format."

## Acknowledgments
A big thank you to [milesial](https://github.com/milesial) for the original [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) implementation. Any modifications made in this fork are built upon the solid foundation they provided.

