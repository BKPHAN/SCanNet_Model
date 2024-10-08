# SCanNet
Pytorch codes of **Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote Sensing Images** [[paper]](https://ieeexplore.ieee.org/document/10443352)


![alt text](https://github.com/ggsDing/SCanNet/blob/main/SCanNet.png)
![alt text](https://github.com/ggsDing/SCanNet/blob/main/L_psd_sc.png)

## Checkpoints

For readers to easily evaluate the accuracy, we provide the trained weights.

SECOND:  

1.[Drive](https://drive.google.com/file/d/1KfA_s3UVqK645WVYPdQ8aIlQkpnuPaPY/view?usp=sharing)  
2.[Baidu](https://pan.baidu.com/s/1zL3H1IlTXB9QnHDxY8sRpg?pwd=SCAN) (pswd: SCAN)

LandsatSCD:  

1.[Drive](https://drive.google.com/file/d/1lCWNUyZyMH7gYTwnhcs4-4oOuveKbJCI/view?usp=drive_link)  
2.[Baidu](https://pan.baidu.com/s/1qih4E1g1c3nbbJ3gFaSlYA?pwd=SCAN) (pswd: SCAN)


## Landsat-SCD

The land-scd dataset needs to be pre-processed to meet the experimental settings in this paper.
More details are provided at [/datasets/LandsatSCD/read_me.md](https://github.com/ggsDing/SCanNet/tree/main/datasets/LandsatSCD)

For readers' convenience, we also provide the preprocessed data:

[Baidu Netdisk](https://pan.baidu.com/s/1ynizp4WST6EeBo6pxo6Kog?pwd=lscd) (psswd lscd)

[Google Drive](https://drive.google.com/file/d/11CkLhakNtfaBH78SGTHxcXKNsBM524H5/view?usp=sharing)

## Cite SCanNet

If you find this work useful or interesting, please consider citing the following BibTeX entry.

```
@article{ding2024joint,
  title={Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote Sensing Images},
  author={Ding, Lei and Zhang, Jing and Guo, Haitao and Zhang, Kai and Liu, Bing and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  year={2024},
  volume={62},
  pages={1-14},
  doi={10.1109/TGRS.2024.3362795}
}
```

(Note: This repository is under construction, contents are not final.)

CUDA:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118