# **Motion Transformer for Unsupervised Image Animation**
## **Codes**

This is the project page of the paper **Motion Transformer for Unsupervised Image Animation (ECCV 2022)**. Due to the security policy of the company, we are in the code approval process, once it is finished, the codes will be released here.

<!-- ### **Updates:** -->
**2022.8.15:** Codes are released.

## **Environments**
The model is trained on 8 Tesla V100 cards, pytorch vesion 1.6 and 1.8 with python 3.6 are tested fine. Basic installations are given in requiremetns.txt.

    pip install -r requirements.txt

## **Datasets**
Following [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) to download **TaiChiHD**,**Voxceleb1**, and **MGIF**. And for the **TED384**, following [MRAA](https://github.com/snap-research/articulated-animation). After downloading and pre-processing, the dataset should be placed in the `./data` folder or you can change the parameter `root_dir` in the yaml config file. Note that we save the video dataset in png frames format for better training IO performance. All train and test video frames are specified in txt files in the `./data` folder. The tree structure of the dataset path is given in the following.

    |-- data/taichi-png, data/TED384-png, data/voxceleb1-png, data/moving-gif-png
        |-- train
            |-- video1
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- video2
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- ...
        |-- test
            |-- video1
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- video2
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- ...

## **Training**
We train the model on 8 cards and use pytorch DistributedDataPrallel.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run.py --config config/dataset.yaml
    
## **Evaluation**
Evaluate video reconstruction with following command, for more metrics, we recommend to see [FOMM-Pose-Evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).

    CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config path/to/config --checkpoint path/to/model.pth  

## **Demo**
To make a demo animation, specify the driving video and source image, the result video will be saved to result.mp4.

    python demo.py --mode demo --config path/to/config --checkpoint path/to/model.pth --driving_video path/to/video.mp4 --source_image path/to/image.png --result_video path/to/result.mp4 --adapt_scale

## **Pretrained models**
Coming soon

## **Animation**
![video](videos/TEDTalks.gif)

## **Citation**
    @inproceedings{tao2022motion,
    title={Motion Transformer for Unsupervised Image Animation},
    author={Tao, Jiale and Wang, Biao and Ge, Tiezheng and Jiang, Yuning and Li, Wen and Duan, Lixin},
    booktitle={European Conference on Computer Vision},
    year={2022}
    }

## **Acknowledgements**
The implemetation is partially borrowed from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [TokenPose](https://github.com/leeyegy/TokenPose) and [TransPose](https://github.com/yangsenius/TransPose), we thank the authors for their excellent works.
