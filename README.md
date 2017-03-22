# TorchPredictorDemo #

### Demo for classifier of face and gender from image or camera using [TorchPredictor](https://github.com/potterhsu/TorchPredictor) with fine-tuned VGG model ###

### Setup ###

* $ mkdir build; cd build
* $ cmake ..
* $ make

### Run ###

* $ ./ImageDemo [Mission] /path/to/Model.tpb /path/to/Image.jpg
* $ ./CameraDemo [Mission] /path/to/Model.tpb /path/to/cascade_face.xml

	> Mission 1 for face classification, 2 for gender classification

### Examples ###

* Image demo for face classification

	```
	$ ./ImageDemo 1 ../vgg_face_model.tpb ../dami_1.jpg
	Parsing model...
	Load image to input...
	Starting forward...
	Output: 
	  Dami = 1
	  Darren = 4.34157e-08
	  Jones = 3.73294e-13
	  Poter = 2.79625e-08
	Result is Dami

	```
	
* Image demo for gender classification

	```
	$ ./ImageDemo 2 ../vgg_gender_model.tpb ../lena_224.jpg
	Parsing model...
	Load image to input...
	Starting forward...
	Output: 
	  Male = 1.65686e-23
	  Female = 1
	Result is Female
	```

* Camera demo for gender classification

	```
	$ ./CameraDemo 2 ../vgg_gender_model.tpb ../haarcascade_frontalface_alt2.xml
	Parsing model...
	Ready
	```



