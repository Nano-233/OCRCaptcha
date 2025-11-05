Here is a professional README.md file for your GitHub portfolio, based on the Jupyter Notebook you provided.

---

# **CRNN CAPTCHA Solver**

This project is an Optical Character Recognition (OCR) model designed to solve alphanumeric CAPTCHAs. It implements a **Convolutional Recurrent Neural Network (CRNN)** architecture combined with a **Connectionist 1Temporal Classification (CTC)** loss function to perform sequence-to-sequence transcription of the CAPTCHA images.

This implementation was developed in Google Colab and includes several key fixes and best practices for Keras 3 / TensorFlow, such as a custom CTC loss function and GPU-based data augmentation.

## **📸 Project Showcase**

Here is an example of the model's predictions on random samples from the test set. The model plots the original image alongside the preprocessed image (which is fed into the network) and shows the "Actual" vs. "Predicted" text.

*(**Note:** You can screenshot the evaluation output from your notebook and add it here. The table below represents the results from the provided run.)*

| Original Image | Preprocessed Image & Prediction |
| :---- | :---- |
| ![Original 0dx9](img/0dx9-0.png) | **Actual:** 0dx9 | **Predicted:** 0dx9 ✓ |
| ![Original w7na](img/w7na-0.png) | **Actual:** w7na | **Predicted:** w7na ✓ |
| ![Original bg2l](img/bg2l-0.png) | **Actual:** bg2l | **Predicted:** b51 ✗ |
| ![Original lag9vt](img/lag9vt-0.png) | **Actual:** lag9vt | **Predicted:** la09vt ✗ |
| ![Original xgyfaku](img/xgyfaku-0.png) | **Actual:** xgyfaku | **Predicted:** xgyfaku ✓ |

### **Performance**

In this run, the model achieved **60.0% (3/5)** accuracy on the random test batch. The final validation loss after 50 epochs was **6.6554**.

This project serves as a strong baseline, and the relatively low accuracy suggests several clear paths for further improvement, highlighting the challenges of OCR on noisy data.

## **🛠️ Technical Details**

This project is more than a simple model import; it involves custom components to handle the specific challenges of CAPTCHA recognition.

### **1\. Advanced Image Preprocessing**

Before being fed to the network, each image undergoes a multi-step preprocessing pipeline using OpenCV to standardize input and remove noise:

1. **Denoising:** A cv2.medianBlur is applied to remove salt-and-pepper noise.  
2. **Contrast Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization) is used to make the characters stand out from the background.  
3. **Binarization:** cv2.adaptiveThreshold converts the image to black and white, isolating the text.  
4. **Artifact Removal:** cv2.morphologyEx with MORPH\_OPEN is used to remove small, non-character artifacts.  
5. **Standardization:** The image is resized to a uniform height (50px) while maintaining its aspect ratio, then padded to a final width of 495px.  
6. **CRNN Input Prep:** The image is transposed to (Width, Height) to be used as a time-sequence for the RNN.

### **2\. Model Architecture: CRNN**

The model is a Convolutional Recurrent Neural Network (CRNN), a standard architecture for sequence-based recognition tasks.

* **Convolutional Base (CNN):** A VGG-style stack of Conv2D, MaxPooling, and BatchNormalization layers acts as a powerful feature extractor. It processes the input image and outputs a sequence of feature vectors.  
* **Recurrent Neck (RNN):** The feature sequence from the CNN is fed into two layers of Bidirectional LSTMs. This allows the model to learn contextual information from the character sequence in both forward and backward directions.  
* **Transcription Head (CTC):** A final Dense layer outputs a probability distribution over all characters in the vocabulary (plus a 'blank' token) for each time-step.

### **3\. Connectionist Temporal Classification (CTC) Loss**

This project uses a custom-implemented CTC loss function (ctc\_loss\_function). CTC is essential for OCR because it solves the problem of not knowing the exact alignment between the input image segments and the output characters. It allows the model to learn to transcribe the text without needing pre-segmented, character-level labels.

### **4\. Keras 3 / TensorFlow Best Practices**

* **GPU-Based Augmentation:** Data augmentation (Rotation, Translation, Zoom) is implemented as a Keras Sequential layer *inside* the model, rather than in the Python generator. This moves the augmentation process onto the GPU, making it significantly more efficient and non-blocking.  
* **Efficient Data Pipeline:** The CaptchaDataGenerator (a keras.utils.Sequence subclass) is used with use\_multiprocessing=True to ensure the data-loading pipeline is parallelized and does not bottleneck the GPU during training.

## **🚀 Future Improvements**

The 10% accuracy provides a clear baseline. The next steps to improve this model would be:

* **Tune Preprocessing:** The current pipeline is aggressive. Experimenting with different thresholding, blurring, and morphological operations could be key to retaining character integrity.  
* **Hyperparameter Tuning:** Adjust the learning rate, batch size, optimizer (e.g., AdamW), and RNN hidden units.  
* **More Augmentation:** Increase the intensity of the data augmentation to make the model more robust to the noise and distortion in the CAPTCHAs.  
* **Architectural Changes:** Experiment with a deeper CNN base (like ResNet) or add an Attention mechanism to the RNN neck, which can help the model focus on the most relevant features.

## **⚙️ How to Run**

1. **Environment:** This notebook is designed for Google Colab with a GPU runtime (T4 recommended).  
2. **Data:**  
   * This project requires a data.zip file containing the CAPTCHA images in main/train/ and main/test/ folders.  
   * The file naming convention for labels is \[label\]-\[id\].png (e.g., 0024miih-0.png).  
   * Place your data.zip file in your Google Drive

   * Alternatively, you can update the DRIVE\_ZIP\_PATH variable in the second code cell to point to your file's location.  
3. **Execute:** Run `notebooks/train.ipynb` notebook cells in order. The script will:  
   * Install dependencies.  
   * Copy and unzip the data from your Drive to the local Colab runtime for faster access.  
   * Preprocess the data and define the model.  
   * Train the model (this will take time) and save the best weights to models/best\_crnn\_model.keras.  
   * Load the best model and run the visual evaluation on 10 test samples.