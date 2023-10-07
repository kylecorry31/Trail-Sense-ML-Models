import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

TF_MODEL_FILE_PATH = 'model_quantized.tflite'
IMG_SIZE = 64
class_names = ['altocumulus', 'altostratus', 'cirrocumulus', 'cirrostratus', 'cirrus', 'clear', 'cumulonimbus', 'cumulus', 'nimbostratus', 'stratocumulus', 'stratus']
cloud_dir = 'clouds'

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

classify_lite = interpreter.get_signature_runner('serving_default')

while True:
    # Choose a random class name
    class_name = class_names[np.random.randint(0, len(class_names))]
    files = os.listdir(cloud_dir + '/' + class_name)
    if len(files) == 0:
        continue
    # Choose a random image from the class
    img_path = cloud_dir + '/' + class_name + '/' + np.random.choice(files)
    # Load the image
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb'
    )

    # Convert the image to a numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand the dimensions of the array to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

    top_3_with_scores = np.argsort(score_lite)[0][::-1][:3]
    top_3 = np.argsort(score_lite)[0][::-1][:3]

    print("Top 3 predictions:")
    for i in range(3):
        print("Class: {}, Score: {:.2f}".format(class_names[top_3[i]], score_lite[0][top_3_with_scores[i]]))
    print("Actual class: {}".format(class_name))

    # Show the image using matplotlib
    plt.imshow(img)
    plt.show()
