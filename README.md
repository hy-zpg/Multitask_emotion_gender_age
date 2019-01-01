# Train_with_mixed_batch_keras

Train a multi-task learning model to estimate age, gender and emotion of an input face. 
The trianing is done from multiple datasets, with a mixed-batch strategy. The code is written in Keras with Tensorflow as the backend. 

The labels for age is from [LAP dataset](http://chalearnlap.cvc.uab.es). 
The labels for gender and emotion is from [MTFL dataset](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html). 
The labels for emotion is from [ferplus] and [fer2013]


* adopting central loss in multi-task learning
* single-task traing
* multi-task traing, such as emotion and age recognition 
* multo-label traing, such as MTFL
