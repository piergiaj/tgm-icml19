# Temporal Gaussian Mixture Layer for Videos

This repository contains the code for our [ICML 2019 paper](https://arxiv.org/abs/1803.06316):

    AJ Piergiovanni and Michael S. Ryoo
    "Temporal Gaussian Mixture Layer for Videos"
    in ICML 2019

If you find the code useful for your research, please cite our paper:

        @inproceedings{piergiovanni2018super,
              title={Temporal Gaussian Mixture Layer for Videos},
              booktitle={International Conference on Machine Learning (ICML)},
              author={AJ Piergiovanni and Michael S. Ryoo},
              year={2019}
        }


# Temporal Gaussian Mixture Layer
The core of our approach, the Temporal Gaussian Mixture (TGM) Layer can be found in [tgm.py](tgm.py).

![mg](/examples/mixture-of-gaussians.png?raw=true "mg")

Multiple (M) temporal Gaussian distributions are learned, and they are combined with the learned soft attention weights to form the C temporal convolution filters. L is the temporal length of the filter.

![share](/examples/tgm-shared.png?raw=true "share")

The kernels are applied to each input channel, C<sub>in</sub>, and a 1x1 convolution is applied to combine the C<sub>in</sub> input channels for each output channel, C<sub>out</sub>.

# Activity Detection Experiments
![model overview](/examples/model.png?raw=true "model overview")

To run our pre-trained models:

```python train_model.py -mode joint -dataset multithumos -train False -rgb_model_file models/multithumos/rgb_baseline -flow_model_file models/multithumos/flow_baseline```

We tested our models on the [MultiTHUMOS](http://ai.stanford.edu/~syyeung/everymoment.html), [Charades](http://allenai.org/plato/charades/), and [MLB-YouTube](https://github.com/piergiaj/mlb-youtube) datasets. We provide our trained models in the models directory.

## Results
### Charades

|  Method | mAP (%) |
| ------------- | ------------- |
| Two-Stream + LSTM [1] | 9.6  |
| Sigurdsson et al. [1]  | 12.1  |
| I3D [2] baseline      | 17.22 |
| I3D + 3 temporal conv. | 17.5 |
| I3D + LSTM          | 18.1  |
| I3D + Fixed temporal pyramid | 18.2|
| I3D + Super-events [4] | 19.41 |
| I3D + 3 TGMs  | 20.6 |
| I3D + Super-events [4] + 3 TGMs | **21.8** |

### MultiTHUMOS

|  Method | mAP (%) |
| ------------- | ------------- |
| Two-Stream [3]  | 27.6  |
| Two-Stream + LSTM [3] | 28.1 | 
| Multi-LSTM [3]  | 29.6  |
| I3D [2] baseline | 29.7 |
| I3D + LSTM | 29.9 |
| I3D + 3 temporal conv. | 24.4 |
| I3D + Fixed Temporal Pyramid | 31.2 |
| I3D + Super-events [4] | 36.4 |
| I3D + 3 TGMs | 44.3 |
| I3D + Super-events [4] + 3 TGMs | **46.4** |


### MLB-YouTube

|  Method | mAP (%) |
| ------------- | ------------- |
| I3D [2] baseline | 34.2 |
| I3D + LSTM | 39.4 |
| I3D + Super-events [4] | 39.1 |
| I3D + 3 TGMs | 40.1 |
| I3D + Super-events [4] + 3 TGMs | **47.1** |


# Example Results
![ex](/examples/res.png?raw=true "mg")

The temporal regions classified as various basketball activities from a basketball game video in MultiTHUMOS. Our TMG layers greatly improve performance.

![gif](/examples/charades_example.gif?raw=true "example")


# Requirements

Our code has been tested on Ubuntu 14.04 and 16.04 using python 2.7, [PyTorch](pytorch.org) version 0.3.1 with a Titan X GPU.


# Setup

1. Download the code ```git clone https://github.com/piergiaj/tgm-icml19.git```

2. Extract features from your dataset. See [Pytorch-I3D](https://github.com/piergiaj/pytorch-i3d) for our code to extract I3D features.

3. [train_model.py](train_model.py) contains the code to train and evaluate models.


# Refrences
[1] G.  A.  Sigurdsson,  S.  Divvala,  A.  Farhadi,  and  A.  Gupta. Asynchronous temporal fields for action recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017

[2] J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[3] S. Yeung, O. Russakovsky, N. Jin, M. Andriluka, G. Mori, and L. Fei-Fei. Every moment counts: Dense detailed labeling of actions in complex videos. International Journal of Computer Vision (IJCV), pages 1–15, 2015

[4] A. Piergiovanni  and  M.  S.  Ryoo.  Learning  latent  super-events  to  detect  multiple  activities  in  videos.   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 [arxiv](https://arxiv.org/abs/1712.01938) [code](https://github.com/piergiaj/super-events-cvpr18)
