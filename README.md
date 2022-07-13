# **Motion Transformer for Unsupervised Image Animation**
## **Codes**

This is the project page of the paper "Motion Transformer for Unsupervised Image Animation" (ECCV 2022). Due to the security policy of the company, we are in the code approval process, once it is finished, the codes will be released here.

## **Method**

In our proposed motion transformer, we introduce two types of tokens in our proposed method: i) image tokens formed from patch features and corresponding position encoding; and ii) motion tokens encoded with motion information. Both types of tokens are sent into vision transformers to promote underlying interactions between them through multi-head self attention blocks. By adopting this process, the motion information canbe better learned to boost the model performance. The final embedded motion tokens are then used to predict the corresponding motion keypoints and local transformations.

![image](https://user-images.githubusercontent.com/38600167/178645760-1f1a9d51-cba4-4083-812e-f3a5ed432a80.png)

## **Animation**

![video](videos/TaiChiHD.gif)

![video](videos/TEDTalks.gif)

![video](videos/VoxCeleb.gif)

