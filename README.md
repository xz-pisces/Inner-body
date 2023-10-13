# Inner-body
This repository contains a tensorflow implementation of "Learning to Infer Inner-Body under Clothing from Monocular Video".

[Project Page](http://cic.tju.edu.cn/faculty/likun/projects/Inner-Body/index.html)



<!-- # Requirement -->



# Installation
Create virtual environment

    conda create -n innerbody python=3.6
    conda activate innerbody

Install cuda and cudnn

    conda install cudatoolkit=9.0

Install tensorflow

    pip install tensorflow-gpu==1.12.0

Install dirt:  [https://github.com/pmh47/dirt](https://github.com/pmh47/dirt)

Install other environments

    pip install -r requirements.txt 

Download the neutral SMPL model from [http://smplify.is.tue.mpg.de/](http://smplify.is.tue.mpg.de) and place it in the assets folder (female_smpl.pkl, neutral_smpl.pkl, male_smpl.pkl).
    

Download pre-trained model weights from [here](https://drive.google.com/file/d/1DNHD3xlAaB-eaZze1UF7s2co7Trg-S7m/view?usp=drive_link
) and place them in the SUNET/weights folder.
    


# Quick start

    sh bash.sh

# Data preparation

If you want to process your own data, some pre-processing steps ([MODNET](https://github.com/ZHKKKe/MODNet), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)) are needed:

    1.Remove the background of the image by MODNET and crop it to 512*512.
    2.Run OpenPose body_25 and face keypoint detection on your images.
    3.Run SCHP semantic segmentation on the image and save to one-hot format by channel. (see SUNET/schp_one_hot.py)


# Dataset

The dataset (current version 1.0) can be downloaded from Baidu Netdisk.

### 1.[Synthetic Dataset](https://pan.baidu.com/s/1JCgXxI8EKK1IURv14InGFw?pwd=1m2b).

### 2.Inner-Body Under Clothing Dataset.


# Citation
Please cite the following paper if it helps your research:

    @article{li2022tvcg,
      author = {Xiongzheng Li and Jing Huang and Jinsong Zhang and Xiaokun Sun and Haibiao Xuan and Yu-Kun Lai and Yingdi Xie and Jingyu Yang and Kun Li},
      title = {Learning to Infer Inner-Body under Clothing from Monocular Video},
      booktitle = {IEEE Transactions on Visualization and Computer Graphics},
      year={2022},
    }
   
# Contact
For more questions, please contact lxz@tju.edu.cn

# License
        Software Copyright License for non-commercial scientific research purposes
        Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License

        License Grant
        Licensor grants you (Licensee) personally a single-user, non-exclusive, non-transferable, free of charge right:

        To install the Model & Software on computers owned, leased or otherwise controlled by you and/or your organization;
        To use the Model & Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects;
        Any other use, in particular any use for commercial, pornographic, military, or surveillance, purposes is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artifacts for commercial purposes. The Data & Software may not be used to create fake, libelous, misleading, or defamatory content of any kind excluding analyses in peer-reviewed scientific research. The Data & Software may not be reproduced, modified and/or made available in any form to any third party.

        The Data & Software may not be used for pornographic purposes or to generate pornographic material whether commercial or not. This license also prohibits the use of the Software to train methods/algorithms/neural networks/etc. for commercial, pornographic, military, surveillance, or defamatory use of any kind. By downloading the Data & Software, you agree not to reverse engineer it.

        No Distribution
        The Model & Software and the license herein granted shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only.

        Disclaimer of Representations and Warranties
        You expressly acknowledge and agree that the Model & Software results from basic research, may contain errors, and that any use of the Model & Software is at your sole risk. 
        
# Acknowledgments
The codes of Inner-Body are largely borrowed from [Octopus](https://github.com/thmoa/octopus).

