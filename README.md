# SVHN-Resnet50
# Sample Output Images #

<img src="https://raw.githubusercontent.com/hgautam750/SVHN-Resnet50/master/retinanet-svhn/samples/samples/test10.png">
<img src="https://raw.githubusercontent.com/hgautam750/SVHN-Resnet50/master/retinanet-svhn/samples/samples/test36.png">
<img src="https://raw.githubusercontent.com/hgautam750/SVHN-Resnet50/master/retinanet-svhn/samples/samples/test70.png">
<img src="https://raw.githubusercontent.com/hgautam750/SVHN-Resnet50/master/retinanet-svhn/samples/samples/test82.png">

## Usage for python code

#### 0. Requirement

* python 3.6
* tensorflow 1.10.0
* keras 2.2.4
* keras-retinanet 0.5.0

#### 1. Digit Detection using pretrained weight file

In this project, the pretrained weight file is stored in [resnet50_full.h5](https://drive.google.com/drive/folders/1kKmDqZ1G4TC-OD3IRdTSJA5H4n6ZAouN). Test set evaluation (13068-images) score is ```mAP: 0.8148```

* Download [resnet50_full.h5](https://drive.google.com/drive/folders/1kKmDqZ1G4TC-OD3IRdTSJA5H4n6ZAouN) to the ```project-root/snapshots```

* Run [infer.py]

##### 

Sources Used: 
a) https://github.com/penny4860/retinanet-digit-detector
b) https://github.com/fizyr/keras-retinanet
