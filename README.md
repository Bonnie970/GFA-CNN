# Our Generalizable Face AuthenticationCNN (GFA-CNN) is available at: https://arxiv.org/abs/1901.05602 
This project is created by Tu Xiaoguang (xguangtu@outlook.com). Any questions pls open issues for my project, I will reply quickly. 

To facility the research in the community of face anti-spoofing, we release our test codes (including the model) for GFA-CNN, the training code will be available upon the acceptance of our paper.  

# We add the code for FDA in the folder "FDA_codes"
Usage: Tensorflow-1.12.0, python-2.7 // Download the "fda_model.rar" package at： https://pan.baidu.com/s/1P3_BrVkc0A4Y9wtL72PiAg using the password: "39lk". Put the package under the folder "FDA_codes/scripts", extract it and then type the below commonds in the terminal (you may need to modify the path in the script "evaluate_inFolder_mfsdPhoto.py" and also the path in the txt test-label files): 
```
python evaluate_inFolder_mfsdPhoto.py --allow-different-dimensions --checkpoint fda_model/mfsdPhotoAtck/fns.ckpt --in-path content --out-path outputs/
```
Some results of the FDA:

<img width="450" height="400" src="https://user-images.githubusercontent.com/8948023/55400706-2caa2100-5581-11e9-95cb-52364136410d.png"/>


# We add the test code in the folder "test_codes"
Usage: Tensorflow-1.12.0, python-2.7 // Download the "models.rar" package at： https://pan.baidu.com/s/1Y3VlXPhGUCSRhRSEmRYIWg using the password: "bafq", or https://drive.google.com/file/d/1UWRCoM_Doq9bmh4bmrCf8U0SGfLnZ1GM/view?usp=sharing
Put the package under the folder "models", extract it and then run the script: " 	interTest_anti_photo_merge_mfsd.py". You may need to change the path in this script and also the path in the txt test-label files.

# Acknowledgement
Thanks for the authors in the [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) for making their excellent works publicly available.

# Citation
If you find our code is useful for your research, pls cite our work:

```
@article{tu2019learning,
  title={Learning Generalizable and Identity-Discriminative Representations for Face Anti-Spoofing},
  author={Tu, Xiaoguang and Zhao, Jian and Xie, Mei and Du, Guodong and Zhang, Hengsheng and Li, Jianshu and Ma, Zheng and Feng, Jiashi},
  journal={arXiv preprint arXiv:1901.05602},
  year={2019}
}
```

