## Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites

### CVPR 2022

#### Ngoc Long Nguyen, Jérémy Anger, Axel Davy,  Pablo Arias, and Gabriele Facciolo


-------------------
![Teaser](HDR-DSP-teaser.jpg)

#### Abstract 
Modern Earth observation satellites capture multi-exposure bursts of push-frame images that can be super-resolved via computational means. In this work, we propose a super-resolution method for such multi-exposure sequences, a problem that has received very little attention in the literature. The proposed method can handle the signal-dependent noise in the inputs, process sequences of any length, and be robust to inaccuracies in the exposure times. Furthermore, it can be trained end-to-end with self-supervision, without requiring ground truth high resolution frames, which makes it especially suited to handle real data. Central to our method are three key contributions: i) a base-detail decomposition for handling errors in the exposure times, ii) a noise-level-aware feature encoding for optimal fusion of frames with varying signal-to-noise ratio and iii) a permutation invariant fusion strategy by temporal pooling operators. We evaluate the proposed method on synthetic and real data and show that it outperforms by a significant margin existing single-exposure approaches that we adapted to the multi-exposure case.

[[pdf](https://openaccess.thecvf.com/content/)]  [[supp](https://openaccess.thecvf.com/content/)]


### Dataset description

[Download link. Multi-Exposure Skysat L1A stacks](https://github.com/centreborelli/HDR-DSP-SR/releases/download/v1/hdr-dsp-real-dataset.zip)

It contains 2660 L1A sequences (folder `crop/`) of size 128x128 (uint16).
The length of each sequence varies from 6 to 15 frames.
Each sequence is produced by registering the frames with integer translations. This means that the frames are purposefully not exactly aligned since it would require resampling of the data.

Associated to each sequence, there are the saturation masks (`satmask/`) and exposure ratios (`ratios/`).
The saturation mask for a frame is a boolean mask, equals to 1 where the sample is valid, 0 otherwise (due to saturation).
The exposure ratios correspond to the exposure times per frame in millisecond as given by Planet as metadata. By dividing a sequence by the ratios vector (one float per frame), the radiometry of the resulting sequence should be approximately equalized. However, due to imprecision in these exposure times, this equalization is not perfect.

Refer to the paper and its supplementary material for more details.




