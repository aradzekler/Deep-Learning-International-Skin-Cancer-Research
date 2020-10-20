# International Skin Imaging Collaboration (ISIC) Image Classification with Convolutional Neural Networks
By: Arad Zekler


# Abstract
As part of the International Skin Imaging Collaboration (ISIC) annual challenge in 2018 we trained a deep convolutional networks to classify the ten thousand images into 7 types (a more correct but pessimistic term is categories) of skin cancer: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis (also called Bowen’s disease), Benign keratosis, Dermatofibroma and Vascular lesion. 

# Introduction
Task 3 of the challenge states the follows as its goal: “Submit automated predictions of disease classification within dermoscopic images” and therefore is a great testbed to measure the performance of different CNN architectures. 

# Background Information
A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image and assign importance (weights and biases) to different aspects in the image and later to be able to differentiate one images from the other. This ability can be used to classify the different parts in the image and output a diagnostic (-an accurate classification) to a picture of a specific medical conditions. The main setback in utilizing this technology to all sorts of fields is the lack of real usable data to train on, and there are even fewer datasets which are anonymous and free to use. That is why the ISIC organization was created (with specifically skin cancer related data) and is upholding annually contests in the subject of utilizing different neural network technologies.

# Project Description
This project will utilize a number of different CNN models in trying to optimize the predicting accuracy of the type of cancer of a random skin lesions.

# The Database
In this work we used the database provided by the ISIC organization, called the HAM1000 [3] Dataset that was acquired with different dematoscope types from different anatomical sites (excluding nails and mucosa) from an anonymous historical sample of patients presented for skin cancer screening from different institutions. 
Images were collected with approval of the Ethics Review Committee of University of Queensland (Protocol-No. 2017001223) and Medical University of Vienna (Protocol-No. 1804/2017. 
The database includes 10,015 pictures and the distribution of disease states represent a modified “real world” setting with an over-representation of malignancies. The response data are sets of binary classifications (contained in a CSV file) for each of the 7 disease states, indicating the diagnosis of each input lesion image.


# Experiment Results
The first experiment used a simple CNN architecture (full model in code) with an output sigmoid layer, and reached an accuracy of 0.9205. even now with a basic CNN we are witnessing good results.
In the next experiment a smallerVGGNet architecture [4] was used, which is a smaller version of the VGGNet [5] more suitable to the hardware limitations. utilizing only 3x3 convolutional layers stacked on top of each other with increasing depth, reducing volume size by max pooling layers and fully connected layers at the end prior to the classifier. The experiment reached 0.943 accuracy!

# Conclusions
From utilizing different CNN architectures, it was demonstrated that this technology can and should be used and further developed to the medical field, in out experiments we reached a peak of 0.943 compared to 0.954 of the winning team. However, it is important to note the results are not suitable to achieving real diagnostic, as we already that the subject has some kind of cancer begin with, but this technology opens doors to help dermatologists with additional information that could lead to a successful and early detection.


# References
[1] Ensembling Convolutional Neural Networks for Skin Cancer Classification by Aleksey Nozdryn-Plotnicki, Jordan Yap, and William Yolland, July, 2018.
[2] Dual Path Networks by Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng, July 6 2017.
[3] The HAM10000 dataset, a large collection of multi-sources dermatoscopic images of common pigmented skin lesions by Philipp Tschandl, Cliff Rosendahl, Harald Kittler, 28 Match 2018.
[4] Keras and Convolutional Neural Networks (CNNs), Adrian Rosebrock, April 16, 2018.
[5] Very Deep Convolutional Networks for Large-Scale Image Recognition, Karen Simonyan, Andrew Zisserman, September 4, 2014.

