# IME_layer
Code for our paper：
</br>Iterative Manifold Embedding Layer Learned by Incomplete Data for Large-scale Image Retrieval.
</br>Jian Xu, Chunheng Wang*, Chengzuo Qi, Cunzhao Shi, Baihua Xiao

NOTE:

tools:
</br>1.The python code is based on the python data science platform Anaconda2.
</br>2.The python code is tested on Linux by PyCharm.


data:
</br>3.Firstly, you should download "feature.rar" from URL: https://pan.baidu.com/s/1dFOwxod (passcode: rtik) and uncompress the file "feature.rar" into folder "data". If you can not download or have any question, please contact me(xujian2015@ia.ac.cn).
</br>The features of convolutional layer(Pool5 layer) of VGG16 for Oxford5k and Paris6k datasets are in path "data\feature". 
</br>4.The order of part detectors are in path "data\filter_select".
</br>5.You should uncompress the file "data\gt_files.rar" into current folder "data". The groundtruth for Oxford5k and Paris6k datasets are in path "data\gt_files".


code:
</br>6.Run evaluate.py, the mAP is printed.
</br>7.Run select_filter.py to get the order of part detectors according to variances. 
