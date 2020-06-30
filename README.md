# Classification on Diagnosis Chest X-Ray Images

## Task

Given the dataset, the model is implemented based on a classification task. This classification task have two classes; Normal (0), Pneumonia (1). Model is implemented in deep learning library PyTorch and the model is selected as pretrained model VGGNet16. VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. Model evaluates that networks of increasing depth using an architecture with
very small (3 x 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. 

<img src="/im/vgg1.png" alt="drawing" width="500"/>

<img src="/im/vgg2.png" alt="drawing" width="500"/>

## Paper

- abs: https://arxiv.org/abs/1409.1556
- pdf: https://arxiv.org/pdf/1409.1556.pdf

## Dataset

The normal chest X-ray depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe , whereas viral pneumonia manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs. The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

data:

    ├── test
    │   ├── NORMAL
    │   └── PNEUMONIA
    ├── train
    │   ├── NORMAL
    │   └── PNEUMONIA
    └── val
        ├── NORMAL
        └── PNEUMONIA
## Samples From Training Set and Class Frequency

<p float="left">
  <img src="/im/samples.png" width="600" />
  <img src="/im/freq.png" width="400" /> 
</p>




