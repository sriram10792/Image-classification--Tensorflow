import label_image
import urllib

#image_path="H:\\Articles to share\\Git codes\\Image Classification using Tensorflow\\image_test.jpg"
#image_path=path , s3 path

if image_path.endswith('.png'):
    urllib.urlretrieve(image_path,'test.png')
    answers1=label_image.classification_function('test.png')
    print (answers1)
if image_path.endswith('.jpg'):
    urllib.urlretrieve(image_path,'test.jpg')
    answers1=label_image.classification_function('test.jpg')
    print (answers1)