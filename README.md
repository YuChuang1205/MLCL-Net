### MLCL-Net code  
#### Paper: **Chuang Yu**, Yunpeng Liu*, Shuhang Wu, Zhuhua Hu, Xin Xia, Deyan Lan, Xin Liu. Infrared small target detection based on multiscale local contrast learning networks[J]. Accepted by *Infrared Physics & Technology*. (2022, SCI) ([[paper](https://doi.org/10.1016/j.infrared.2022.104107)])  

#### We also uploaded the complete dataset [[link](https://github.com/YuChuang1205/SIRST-dataset-MLCL-Net-version)] we used.

**1.Note that we have also uploaded the final weights and the final detection results to the "sample" folder, and the results correspond to those in the paper.**  

**2.In "sample" folder, "finally_model" folder includes the finally model; "test_ground_truth_label" folder includes the ground truth labels; "test_image" includes the samples that needs to be tested； "test_results" includes the results after the test samples have been tested by the finally model, and the results correspond to those in the paper.**

**3.We still have some work in progress, and the complete project code will be open sourced in the future.**


<!-- **4. I am open to collaboration opportunities (anytime & anywhere & any type). If you need collaborative research, please email me (yuchuang@sia.cn). Please be sure to state your name, unit, and some representative research works in the email.** -->


When debugging code for others, I found an interesting phenomenon. When the original SIRST dataset is not directly resized to 512×512 pixels, but directly filled with "0" at the bottom right, directly using this code has higher IoU and nIoU. I only experimented it once, and the results were 0.793 and 0.781. We consider that the direct resize method is not very friendly to the target boundary of the original image. However, for a more rigorous comparison, the direct resize method is still used, which is consistent with the existing deep learning-based methods. Because the performance is relative, it is more meaningful to test under the same dataset. When you conduct a comparative experiment on the SIRST dataset, please be sure to use the resize method, or directly use the dataset I made in the link above, so that the results are comparable. 


It is worth mentioning that the LCL module we mentioned is deeply influenced by the idea of PCM. Our purpose is to build a simple and efficient end-to-end full convolution infrared small target detection network. there are still a little differences between LCL and PCM. After all, the deep learning network structure has its own characteristics. But we consider the LCL module to be an approximate generalized representation of PCM. Compared with ANN, the essence of convolutional operation is to learn the relationship between local pixels. The designed LCL module can more clearly highlight the idea of local contrast. At the same time, the (M)LCL module effect is obvious.





