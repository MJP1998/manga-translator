# Personal project : Manga Translator

This project is a personnal project from November 2020 with the goal of automating manga translation with the object detection algorithms : YOLOv4 and YOLOv3. <br/>
The final goal of this little project was to allow the use this model in a website with tensorflow-js by reimplementing the different models but after a few tries I understood that the computing power demanded to each computer was way too big and therefore the tensorflow-js implementation was not adapted. <br/>  

It is a personal project done outside of courses hours to self-study on the potential and the difficulty of implementation of the different models before I had any official courses on the subject.

## Implementation details

We made use of Roboflow's tools to use Yolov4 on a custom dataset, we then tried yolov3 to reduce the amount of parameters but neither of these worked with tensorflow-js.
I created a 300 images dataset for text detection.

To translate a full chapter, we used the following pipeline :
1. Downloaded the images from the website in input
2. Detected the text zone with the Yolo model
3. Detected the text inside with easy-ocr
4. Translated it with Google translate api
5. Inpainted the zone of the text
6. Put the translated text inside with the right dimensions 

## Result

With the Yolov4-tiny model, the result is quite pleasant, even if the translation is still lacking, the detection is pretty accurate on many samples. 
Yolov4 and v3 perform quite well, we show some of the results below.
To load the model in a simple manner, we converted it into a keras model and then loaded it thanks to a yolov4 implementation taken form this repositary : <a src="https://github.com/taipingeric/yolo-v4-tf.keras">YoloV4 Keras</a>


<br/>  

Detected         |  Translated
:-------------------------:|:-------------------------:
<img style= "display: inline-block; margin: 10px"  src="https://user-images.githubusercontent.com/64918024/137822734-2dcc55d4-f0fb-48d6-8745-f2373b40be90.png" alt="example_1_before" width="200" height = "300"/> | <img style= "display: inline-block; margin: 10px"  src="https://user-images.githubusercontent.com/64918024/137823212-2a467ba1-38ca-4e9e-8c03-cd4c8fa8a936.png" alt="example_1_after" width="200" height="300"/>
<img style= "display: inline; margin: 10px" src="https://user-images.githubusercontent.com/64918024/137822752-faeb25dd-1af1-4c13-a7dc-66e83c923679.png" alt="example_2_before" width="200" height = "300"/> | <img style= "display: inline; margin: 10px" src="https://user-images.githubusercontent.com/64918024/137823338-b6d7b492-ab0d-4568-b3aa-60adc03d78b1.png" alt="example_2_after" width="200" height="300"/>
<img style= "display: inline-block; margin: 10px" src="https://user-images.githubusercontent.com/64918024/137822775-c83b5897-7452-4f31-9eb9-68d9c0d4e9ac.png" alt="example_3_before" width="200" height = "300"/> | <img style= "display: inline-block; margin: 10px"  src="https://user-images.githubusercontent.com/64918024/137823348-25cbf3c6-477a-48ee-9150-d2f1b40be0de.png" alt="example_3_after" width="200" height="300"/>

<br/>  


Thanks to Darknet team, Roboflow and Google Colab for making such a custom use of existing algorithms nice and easy. 




