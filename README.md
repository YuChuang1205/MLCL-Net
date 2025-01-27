### MLCL-Net code    
#### Paper: **Chuang Yu**, Yunpeng Liu*, Shuhang Wu, Zhuhua Hu, Xin Xia, Deyan Lan, Xin Liu. Infrared small target detection based on multiscale local contrast learning networks[J]. Accepted by *Infrared Physics & Technology*. 2022. ([[paper](https://doi.org/10.1016/j.infrared.2022.104107)])  

### !!! For the training and testing of MLCL-Net, we have integrated it into our complete framework demo ([[link](https://github.com/YuChuang1205/MSDA-Net)]) for direct use.

#### We also uploaded the complete dataset [[link](https://github.com/YuChuang1205/SIRST-dataset-MLCL-Net-version)] we used.  

**1.Note that we have also uploaded the final weights and the final detection results to the "sample" folder, and the results correspond to those in the paper.**  

**2.In "sample" folder, "finally_model" folder includes the finally model; "test_ground_truth_label" folder includes the ground truth labels; "test_image" includes the samples that needs to be tested； "test_results" includes the results after the test samples have been tested by the finally model, and the results correspond to those in the paper.**

**3.In addition, for small objects with large sizes in some datasets, you can also choose to fuse LCL(7), LCL(9) and so on. I have commented them out in the code. For details, you can look at "model.py".**


### Answers to questions asked by some researchers

**For MLCL-Net, will there be a large fluctuation in the results?**  
**Reply:** Due to the small number of SIRST data samples and many scenes, the results of most deep learning methods will fluctuate to a certain extent without any reasonable training strategy. You need to use some reasonable training strategies (such as DA) to make the results as stable as possible. The multiple experimental results of our project show that the fluctuation is about 0.01 (The fluctuations of the results of other ablation experiments in the paper are basically the same.). The complete project code will be open sourced in the future. At the same time, our latest work **(ALCL-Net)** ([[paper](https://ieeexplore.ieee.org/document/9785618)]) ([[code](https://github.com/YuChuang1205/ALCL-Net)]) and **(GGL-Net)** ([[paper](https://ieeexplore.ieee.org/abstract/document/10230271)]) is more stable and efficient. 


**Data?**  
When debugging code for others, I found an interesting phenomenon. When the original SIRST dataset is not directly resized to 512×512 pixels, but directly filled with "0" at the bottom right, directly using this code has higher IoU and nIoU. I only experimented it once, and the results were 0.793 and 0.781. We consider that the direct resize method is not very friendly to the target boundary of the original image. However, for a more rigorous comparison, the direct resize method is still used, which is consistent with the existing deep learning-based methods. Because the performance is relative, it is more meaningful to test under the same dataset. When you conduct a comparative experiment on the SIRST dataset, please use the resize method, or directly use the dataset I made in the link above, so that the results are comparable. 


### Other supplements
It is worth mentioning that the LCL module we mentioned is deeply influenced by the idea of PCM. Our purpose is to build a simple and efficient end-to-end full convolution infrared small target detection network. there are still a little differences between LCL and PCM. After all, the deep learning network structure has its own characteristics. But we consider the LCL module to be an approximate generalized representation of PCM. Compared with ANN, the essence of convolutional operation is to learn the relationship between local pixels. The designed LCL module can more clearly highlight the idea of local contrast. At the same time, the (M)LCL module effect is obvious.  



