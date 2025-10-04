# CS 4243 Project Plan: CRNN for CAPTCHA Recognition

## 1. Preprocessing Pipeline
Our preprocessing function will perform the following steps, inspired by the Keras OCR example, to handle variations in character brightness:
1.  Read the image and convert it to grayscale.
2.  Resize the image to a fixed height (50px) while maintaining aspect ratio.
3.  Pad the image to a fixed width (200px).
4.  Normalize pixel values to the [0, 1] range.
5.  Transpose the image so the shape is (width, height, 1) for the RNN.

## 2. Model Architecture (CRNN)
Our model will consist of three main blocks:
-   **CNN Backbone:** Two blocks of `Conv2D` and `MaxPooling2D` layers to extract features.
-   **RNN Backbone:** Two layers of `Bidirectional(LSTM(...))` to read the sequence of features and learn contextual patterns.
-   **Output Layer:** A `Dense` layer with `softmax` activation to output character probabilities for the CTC loss function.

## 3. Loss Function
We will use a custom `CTCLayer`. Its role is to calculate the CTC loss, which allows the model to be trained on unsegmented text by automatically aligning the RNN output with the ground truth labels.