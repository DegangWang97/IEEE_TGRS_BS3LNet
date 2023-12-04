# BS<sup>3</sup>LNet
This is the official repository for  ["BS<sup>3</sup>LNet: A New Blind-Spot Self-Supervised Learning Network for Hyperspectral Anomaly Detection"](https://ieeexplore.ieee.org/abstract/document/10049187) in IEEE Transactions on Geoscience and Remote Sensing (TGRS). 

![alt text](./figs/BS3LNet_model.jpg)

## Abstract

Recent years have witnessed the flourishing of deep learning-based methods in hyperspectral anomaly detection (HAD). However, the lack of available supervision information persists throughout. In addition, existing unsupervised learning/semisupervised learning methods to detect anomalies utilizing reconstruction errors not only generate backgrounds but also reconstruct anomalies to some extent, complicating the identification of anomalies in the original hyperspectral image (HSI). In order to train a network able to reconstruct only background pixels (instead of anomalous pixels), in this article, we propose a new blind-spot self-supervised learning network (called BS<sup>3</sup>LNet) that generates training patch pairs with blind spots from a single HSI and trains the network in self-supervised fashion. The BS<sup>3</sup>LNet tends to generate high reconstruction errors for anomalous pixels and low reconstruction errors for background pixels due to the fact that it adopts a blind-spot architecture, i.e., the receptive field of each pixel excludes the pixel itself and the network reconstructs each pixel using its neighbors. The above characterization suits the HAD task well, considering the fact that spectral signatures of anomalous targets are significantly different from those of neighboring pixels. Our network can be considered a superb background generator, which effectively enhances the semantic feature representation of the background distribution and weakens the feature expression for anomalies. Meanwhile, the differences between the original HSI and the background reconstructed by our network are used to measure the degree of the anomaly of each pixel so that anomalous pixels can be effectively separated from the background. Extensive experiments on two synthetic and three real datasets reveal that our BS<sup>3</sup>LNet is competitive with regard to other state-of-the-art approaches.


## Setup

### Requirements

Our experiments are done with:

- Python 3.9.12
- PyTorch 1.12.1
- numpy 1.21.5
- scipy 1.7.3
- torchvision 0.13.1

## Prepare Dataset

Put the data(.mat [data, map]) into ./data

## Training and Testing

### Training
```shell
python main.py --command train --dataset AVIRIS-II --batch_size 100 --epochs 1000 --learning_rate 1e-4 --patch 19 --ratio 0.9 --gpu_ids 0
```

### Testing
```shell
python main.py --command predict --dataset AVIRIS-II --batch_size 100 --epochs 1000 --learning_rate 1e-4 --patch 19 --ratio 0.9 --gpu_ids 0
```

- If you want to Train and Test your own data, you can change the input dataset name (dataset) and tune the parameters, such as Learning rate (learning_rate), Patch size (patch), and Candidate pixel ratio (1-ratio).

## Citation

If the work or the code is helpful, please cite the paper:

```
@article{gao2023bs3lnet,
  author={Gao, Lianru and Wang, Degang and Zhuang, Lina and Sun, Xu and Huang, Min and Plaza, Antonio},
  journal={IEEE Trans. Geosci. Remote Sens.}, 
  title={{BS$^{3}$LNet}: A New Blind-Spot Self-Supervised Learning Network for Hyperspectral Anomaly Detection}, 
  year={2023},
  volume={61},
  pages={1-18},
  DOI={10.1109/TGRS.2023.3246565}
  }
```

## Acknowledgement

The codes are based on [Noise2Void](https://github.com/hanyoseob/pytorch-noise2void). Thanks for their awesome work.

## Contact
For further questions or details, please directly reach out to wangdegang20@mails.ucas.ac.cn
