Plant Species Classification with MobileNetV2
This project classifies medicinal plant species using MobileNetV2. The model is fine-tuned on a custom dataset of segmented leaf images.

Features
MobileNetV2 pre-trained model with custom classification layers.
Data augmentation: Rotation, shifting, zooming, and flipping.
Visualization: Displays sample images from each class (species).

How to Run
Clone the repository and mount your dataset (if using Google Colab):
  python
  Copy code
    from google.colab import drive
    drive.mount('/content/drive')
  Train the model:
  python
  Copy code
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)
    Predict plant species from a leaf image:
  python
  Copy code
    predicted_label, confidence = predict_plant(image_path, label_mapping)
    print(predicted_label, confidence)
  Requirements
    TensorFlow 2.x
    Python 3.x
    Matplotlib, Numpy
