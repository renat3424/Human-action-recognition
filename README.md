# Human action recognition 
## Neural network model for human action recognition written with python and pytorch



This is a program of neural network model based on InceptionV3 convolutional neural network and Transformer architecture based model. The InceptionV3 model is used for feature extraction from sequence of video frames and transformer model is used for classification of obtained feature sequence. The model was trained on a subset of UCF101 dataset which contains next classes: CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing. The accuracy of a model is 96 percent on test data. Used subset for model can be downloaded at: https://drive.google.com/file/d/1vR-6aWMso_sx92DMnUaI8jf3arRfdSn4/view?usp=sharing. The entire dataset can be downloaded at official website of ucf center for research in computer vision: https://www.crcv.ucf.edu/data/UCF101.php. It is better to first preprocess entire data and get features into files and after that using these data files train a model as preprocessing takes large amount of time.

