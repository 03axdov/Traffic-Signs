A Neural Network for detecting german traffic signs on the GTSRB dataset. Will later be used for a basic self-driving car simulation

Using Tensorflow and will implement yoloV4 for object detection once the model for detecting traffic signs, cars etc is created.

I will implement code for keeping track of the model's detections, having a 'speed' variable that will be set to the current speed limit, unless there is a stop sign or something akin to one.

The model has an accuracy of roughly 99%. Load it by calling tensorflow.keras.models.load_model("path-to-model") <--- Path to model would be "./Models" in this case
