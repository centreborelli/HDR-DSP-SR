# Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites (CVPR 2022)

[HDR-DSP](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Self-Supervised_Super-Resolution_for_Multi-Exposure_Push-Frame_Satellites_CVPR_2022_paper.pdf) is the first joint super-resolution and HDR neural network for push-frame satellites. HDR-DSP can be trained on real data thanks to self-supervised learning.

### Quick start

1. Install pytorch and torchvision

`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`

2. Download the SkySat multi-exposure data [here](https://github.com/centreborelli/HDR-DSP-SR/releases/download/v1/hdr-dsp-real-dataset.zip).

`wget https://github.com/centreborelli/HDR-DSP-SR/releases/download/v1/hdr-dsp-real-dataset.zip`
`unzip hdr-dsp-real-dataset.zip`

3. Preprocess the data using the notebook RemoveSaturation.ipynb to remove saturated frames and to categorize sequences by their length.

### Training

The command 

```python train.py```

launches the training the HDR-DSP super-resolution network (see `train.py` file for more options). It requires pre-trained weights for the motion estimation sub-network stored in a file  `pretrained_Fnet.pth.tar`. We provide our pre-trained weights, but if you want to train it yourself you can do it with the command:

```python train_FNet.py```

### Testing

`python test.py`
