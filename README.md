# Classification on Diagnosis Chest X-Ray Images

## Task

Given the dataset, the model is implemented based on a classification task. This classification task have two classes; Normal (0), Pneumonia (1). Model is implemented in deep learning library PyTorch and the model is selected as pretrained model VGGNet16. VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. Model evaluates that networks of increasing depth using an architecture with
very small (3 x 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. 

<img src="/im/vgg1.png" alt="drawing" width="600"/>

<img src="/im/vgg2.png" alt="drawing" width="600"/>
