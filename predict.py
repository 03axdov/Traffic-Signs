import tensorflow as tf
import numpy as np

def predict(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60,60])
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    return np.argmax(predictions)


if __name__ == "__main__":

    model = tf.keras.models.load_model("./Models")
    img_path = ""

    predict(model=model, img_path=img_path)