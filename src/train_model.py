# ==============================================================================
# CS 4243: CAPTCHA Recognition Final Script (Train & Evaluate)
# ==============================================================================
import os
import glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import argparse
import sys
from tqdm import tqdm

# --- 1. SETUP AND CONFIGURATION ---

# Paths and Parameters
DATA_DIR = "data/main/"
IMG_WIDTH = 495
IMG_HEIGHT = 50
BATCH_SIZE = 16
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.0005

# --- Argument Parser for Mode Selection ---
parser = argparse.ArgumentParser(description="Train and Evaluate CAPTCHA Model")
parser.add_argument(
    '--mode',
    type=str,
    default='full_run',
    choices=['train', 'evaluate', 'full_run'],
    help="Script mode: 'train' only, 'evaluate' only, or 'full_run' (train then evaluate)."
)
parser.add_argument(
    '--model_type',
    type=str,
    default='base',
    choices=['base', 'deep'],
    help="Model architecture for training: 'base' (2-layer) or 'deep' (4-layer)."
)
parser.add_argument(
    '--model_path',
    type=str,
    help="Path to the .keras model file (required for 'evaluate' mode)."
)
args = parser.parse_args()


# --- 2. DATA LOADING AND VOCABULARY CREATION ---

def load_data(data_dir):
    """Loads image paths and labels from the specified directory."""
    train_image_paths = sorted(glob.glob(os.path.join(data_dir, "train/*.png")))
    train_labels = [os.path.basename(p).split("-")[0] for p in train_image_paths]
    valid_image_paths = sorted(glob.glob(os.path.join(data_dir, "test/*.png")))
    valid_labels = [os.path.basename(p).split("-")[0] for p in valid_image_paths]

    all_labels = train_labels + valid_labels

    if not all_labels:
        print(f"Error: No images found in {data_dir}. Check DATA_DIR path.")
        sys.exit(1)

    characters = sorted(list(set(char for label in all_labels for char in label)))
    max_length = max([len(label) for label in all_labels])

    print(f"Number of training images: {len(train_image_paths)}")
    print(f"Number of validation images: {len(valid_image_paths)}")
    print(f"Number of unique characters: {len(characters)}")
    print(f"Characters present: {''.join(characters)}")
    print(f"Maximum label length: {max_length}")

    return train_image_paths, train_labels, valid_image_paths, valid_labels, characters, max_length


# Load global constants
train_paths, train_labels, valid_paths, valid_labels, characters, max_length = load_data(DATA_DIR)

# --- 3. DATA PIPELINE (GLOBAL) ---

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
padding_value = char_to_num.vocabulary_size()


def encode_single_sample(img_path, label):
    """Encodes a single image-label pair into tensors."""
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    label_tensor = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label_tensor}


padding_shapes = {
    "image": tf.TensorShape([IMG_WIDTH, IMG_HEIGHT, 1]),
    "label": tf.TensorShape([max_length]),
}


def create_dataset(paths, labels, batch_size=BATCH_SIZE):
    """Creates a tf.data.Dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = (
        dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(
            batch_size,
            padded_shapes=padding_shapes,
            padding_values={"image": 0.0, "label": tf.cast(padding_value, dtype=tf.int64)}
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


# --- 4. CTC LOSS LAYER (GLOBAL) ---

class CTCLayer(layers.Layer):
    """Custom Keras layer to compute CTC loss."""

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.nn.ctc_loss

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        label_length = tf.cast(tf.math.count_nonzero(tf.not_equal(y_true, padding_value), axis=1), dtype="int32")
        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
        loss = self.loss_fn(y_true, y_pred, label_length, input_length, logits_time_major=False)
        self.add_loss(loss)
        return y_pred




def build_model(model_type='base'):
    """Builds the CRNN model."""
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")

    # CNN Backbone
    if model_type == 'deep':
        print("Building DEEP 4-layer CNN model...")
        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        new_shape_w = IMG_WIDTH // 16
        new_shape_h = IMG_HEIGHT // 16
        new_shape = (new_shape_w, new_shape_h * 256)

    else:  # 'base' model
        print("Building BASE 2-layer CNN model...")
        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        new_shape_w = IMG_WIDTH // 4
        new_shape_h = IMG_HEIGHT // 4
        new_shape = (new_shape_w, new_shape_h * 64)

    # Reshape for RNN
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # RNN Backbone
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output Layer for CTC
    x = layers.Dense(char_to_num.vocabulary_size() + 1, name="dense_output")(x)

    # Add CTC layer
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name=f"ocr_captcha_model_{model_type}")

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model


# --- 6. DECODING FUNCTIONS (for evaluation) ---

def ctc_decode(y_pred):
    """Decodes the output of the CTC layer."""
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    results = keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]
    return results


def clean_string(s):
    """Removes the padding token '[UNK]' from a decoded string."""
    return s.replace('[UNK]', '')


def decode_batch_predictions(preds):
    """Decodes a batch of predictions into strings."""
    decoded_preds = ctc_decode(preds)
    output_text = []
    for res in decoded_preds:
        res_str = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(clean_string(res_str))
    return output_text


def labels_to_strings(labels):
    """Converts a batch of label tensors into strings."""
    output_text = []
    for res in labels:
        res_str = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(clean_string(res_str))
    return output_text


# --- 7. FUNCTIONAL BLOCKS ---

def train_model(model_type):
    """Trains a new model."""
    print(f"\n--- Starting training for {model_type} model ---")

    # Create the datasets
    train_dataset = create_dataset(train_paths, train_labels)
    validation_dataset = create_dataset(valid_paths, valid_labels)

    # Build the model
    model = build_model(model_type=model_type)
    model.summary()

    # Setup callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )

    model_save_path = f"src/captcha_model_{model_type}.keras"
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint],
    )

    print(f"\n--- Training Complete ---")
    print(f"Best model saved to '{model_save_path}'")
    return model_save_path


def evaluate_model(model_path):
    """Evaluates a saved .keras model."""
    print(f"--- Evaluating model: {model_path} ---")

    # 1. Load the test data
    # We unbatch and re-batch by 1 for exact accuracy calculation
    test_dataset = create_dataset(valid_paths, valid_labels, batch_size=1)
    print(f"Loaded {len(valid_paths)} test samples.")

    # 2. Load the trained model
    custom_objects = {"CTCLayer": CTCLayer}
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"\n---!! ERROR !! ---")
        print(f"Failed to load model from {model_path}")
        print(f"Error details: {e}\n")
        return

    # 3. Create a prediction model
    try:
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense_output").output
        )
    except Exception as e:
        print(f"\n---!! ERROR !! ---")
        print("Failed to create prediction model. Layer names 'image' or 'dense_output' not found.")
        print(f"Error details: {e}\n")
        return

    # 4. Loop through the dataset and evaluate
    total_samples = 0
    correct_captcha = 0
    total_char_errors = 0
    total_chars = 0

    for batch in tqdm(test_dataset, desc="Evaluating"):
        images, labels = batch["image"], batch["label"]

        preds = prediction_model.predict_on_batch(images)
        pred_texts = decode_batch_predictions(preds)
        orig_texts = labels_to_strings(labels)

        for i in range(len(orig_texts)):
            total_samples += 1
            if pred_texts[i] == orig_texts[i]:
                correct_captcha += 1

            distance = tf.edit_distance([orig_texts[i]], [pred_texts[i]], normalize=False).numpy()[0]
            total_char_errors += distance
            total_chars += len(orig_texts[i])

    # 5. Print the final metrics
    if total_samples == 0:
        print("No samples were evaluated.")
    else:
        captcha_accuracy = (correct_captcha / total_samples) * 100
        cer = (total_char_errors / total_chars) * 100
        char_accuracy = 100 - cer

        print("\n--- Evaluation Complete ---")
        print(f"Model: {model_path}")
        print(f"Total Samples: {total_samples}")
        print("\n--- Project Metrics ---")
        print(f"Captcha Recognition Accuracy (Exact Match): {captcha_accuracy:.2f}%")
        print(f"Character Recognition Accuracy (100 - CER): {char_accuracy:.2f}%")
        print(f"  (Character Error Rate: {cer:.2f}%)")


# --- 8. MAIN EXECUTION ---

if __name__ == "__main__":

    if args.mode == 'train':
        if not args.model_type:
            print("Error: --model_type ('base' or 'deep') is required for 'train' mode.")
        else:
            train_model(args.model_type)

    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path is required for 'evaluate' mode.")
        else:
            evaluate_model(args.model_path)

    elif args.mode == 'full_run':
        if not args.model_type:
            print("Error: --model_type ('base' or 'deep') is required for 'full_run' mode.")
        else:
            print(f"--- Starting Full Run for {args.model_type} model ---")
            saved_model_path = train_model(args.model_type)
            evaluate_model(saved_model_path)