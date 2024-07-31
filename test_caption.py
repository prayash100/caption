import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from pickle import load
from PIL import Image
import os

# Define the custom function
def custom_not_equal(x):
    return tf.cast(tf.not_equal(x[0], x[1]), tf.float32)

# Load the tokenizer
try:
    with open(r'C:\Users\PRATYUSH\Desktop\caption\tokenizer.p', 'rb') as f:
        tokenizer = load(f)
except FileNotFoundError:
    print("Tokenizer file 'tokenizer.p' not found.")
    exit()

# Load the trained model
try:
    model = load_model(
        r'C:\Users\PRATYUSH\Desktop\caption\models\my_model.h5',
        custom_objects={'NotEqual': tf.keras.layers.Lambda(custom_not_equal)}
    )
except FileNotFoundError:
    print("Model file 'my_model.h5' not found.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load and preprocess the image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        exit()

# Generate a caption for an image
def predict_caption(image_path, model, tokenizer, max_length):
    # Extract features
    try:
        xception_model = Xception(include_top=False, pooling='avg')
        image = preprocess_image(image_path)
        feature = xception_model.predict(image)
    except Exception as e:
        print(f"Error extracting features: {e}")
        exit()
    
    # Generate caption
    in_text = 'startseq'
    for _ in range(max_length):
        try:
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            y_pred = model.predict([feature, sequence], verbose=0)
            y_pred = np.argmax(y_pred)
            word = tokenizer.index_word.get(y_pred)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        except Exception as e:
            print(f"Error during prediction: {e}")
            break
    
    return ' '.join(in_text.split()[1:-1])  # Remove 'startseq' and 'endseq'

# Example usage
if __name__ == "__main__":
    image_path = r'C:\Users\PRATYUSH\Pictures\44856031_0d82c2c7d1.jpg'
    max_length = 33  # Ensure this matches the max_length used during training
    if not os.path.isfile(image_path):
        print(f"Image file '{image_path}' not found.")
    else:
        caption = predict_caption(image_path, model, tokenizer, max_length)
        print("Generated Caption:", caption)
