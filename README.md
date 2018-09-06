# Image-classification--Tensorflow
Image classification using tensorflow for poets


Image Classification using TensorFlow

Train a simple image classifier using transfer learning. Transfer learning is the method of reusing the already trained model, and retraining it to our problem. This is short and effective

This method uses the model trained on ImageNet data.



Requirements

1)Python
2)Tensorflow package

For windows

pip install --upgrade tensorflow

To train the model, clone the repository, open sourced by google.
https://github.com/googlecodelabs/tensorflow-for-poets-2

To clone,

git clone https://github.com/googlecodelabs/tensorflow-for-poets-2



Now we need the training images to retrain the model. To do this, manually classify the images in seperate folders in different folders, as shown in the image
Each folder should contain several hundred images preferably for better learning


The retraining is done on MobileNET, a small CNN. CNN performs repeated calculation at the same point on the image which increases the accuracy and reduces the loss function.


"The following lines are advised to be run in bash shell, since some commands span more than a line which causes an issue in normal command prompt"

Higher the size of the images, better is the learning, more is the efficiency, but the processing time is more too.

IMAGE_SIZE=224

ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"


To retrain the model, it is required to run the  "retrain.py" file in the scripts folder that was a part of the clone
Several parameters in the retrain can be changed according to our requirements.
Batch size, learning rate, training steps,testing percentage etc are some of them
This code is reusable and needs no change other the small

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/train_photos


Note: The train_photos folder contains the folder structure that was created before hwich contains the different categories of classifications that is required


The script downlaods the pre-trained model, adds a new final layer, trains the layer on the classification that is given by us. By using this IMAGENET model, we are giving the pretrained information as the input to the final layer in the network.

Options about bottlenecks and tensorflow diagnosis can be added by referencing from
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3



The  "retrain.py " creates information in the two files "retrained_graph.pb" and "retrained_labels.txt" inside "tf_files" folder.

Now when an image path is given as the argument to the label_image file, the classification of the image is displayed

The code includes "testing_file.py" from where the path can be added or modified. The entire label_image.py file is written as a function and imported into this "testing_file.py"

To test a class of the new image, pass the image path into this file, or add an argument parser to accept file path as an argument during run time.

The "label_image.py" is written as a function since the objective was to deploy this into AWS lambda function. But the lambda function requires the dependent packages to be zipped along with the python file. The tensorflow package was beyond the space limits for a lambda function which hindered the deployment. The code to deploy in ec2 is also attached. Note " The code to run in ec2 is written in python 2 ". That code can accept an s3 package and the same can be deployed in a lambda too with slight modification

The result for this classification is attached.

