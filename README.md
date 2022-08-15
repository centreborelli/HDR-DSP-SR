# Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites (CVPR 2022)

[HDR-DSP](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Self-Supervised_Super-Resolution_for_Multi-Exposure_Push-Frame_Satellites_CVPR_2022_paper.pdf) is the first joint super-resolution and HDR neural network for push-frame satellites. HDR-DSP can be trained on real data thanks to self-supervised learning.

### Quick start

1. Install pytorch and torchvision
2. Download the SkySat multi-exposure data [here](https://github.com/centreborelli/HDR-DSP-SR/releases/download/v1/hdr-dsp-real-dataset.zip).
3. Preprocess the data using the notebook RemoveSaturation.ipynb to remove saturated frames and to categorize sequences by their length.

### Training

`python train.py`

### Testing

`python test.py`
