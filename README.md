# MoodAnalyzer

# Text Sentiment Classification Using MLP

This repository contains a project that applies **Deep Learning** techniques to classify textual data into sentiment categories. The focus of this project is on classifying textual posts into predefined classes using a Multi-Layer Perceptron (MLP) model.

## Features

- **Text Feature Extraction**: Uses TF-IDF for extracting meaningful text features.
- **Categorical and Numerical Data Processing**: Handles multi-type data with One-Hot Encoding and Standardization.
- **Class Balancing**: Applies SMOTE for oversampling underrepresented classes.
- **Deep Learning**: Implements a fully connected MLP model with customizable layers and dropout.
- **Binary and Multi-Class Support**: Supports both binary classification (e.g., Positive vs. Negative + Neutral) and multi-class classification.
- **Visualization:** Displays a confusion matrix and classification metrics for model evaluation.

## Dataset
https://www.kaggle.com/code/ramakrushnamohapatra/social-media-sentiment-analysis/input
The dataset contains:
- **Post Content**: Textual data representing user posts.
- **Sentiment**: Labels representing the sentiment of each post (e.g., Negative, Neutral, Positive).
- **Other Features**: Additional categorical and numerical features such as language, post type, and engagement metrics.

  **Note:** Replace `synthetic_social_media_data.csv` with your actual dataset.

Example structure of the dataset:

| Post Content             | Sentiment | Post Type | Language | Number of Likes | Number of Shares |
|--------------------------|-----------|-----------|----------|-----------------|------------------|
| "Great product!"         | Positive  | text      | en       | 150             | 20               |
| "Not worth it."          | Negative  | text      | en       | 45              | 5                |

## Model Architecture

The model is a Multi-Layer Perceptron (MLP) consisting of:

1. **Input Layer**: Accepts the processed features.
2. **Hidden Layers**: Fully connected layers with ReLU activation and Dropout for regularization.
3. **Output Layer**: 
   - For binary classification: 1 neuron with a Sigmoid activation.
   - For multi-class classification: 3 neurons with a Softmax activation.

### Example Architecture

```plaintext
Input Layer (Feature Size: N)
    Dense Layer (64 units, ReLU activation)
    Dropout (0.6)
    Dense Layer (32 units, ReLU activation)
    Dropout (0.6)
Output Layer (Binary: 1 unit with Sigmoid | Multi-Class: 3 units with Softmax)
```

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Open the project in Google Colab or a Python environment with the following dependencies:
   - `pandas`
   - `numpy`
   - `sklearn`
   - `seaborn`
   - `matplotlib`
   - `tensorflow`
   - `imblearn`

## Usage

1. **Preprocess the Dataset**:
   - Extract and preprocess textual features using TF-IDF.
   - Process categorical and numerical features with One-Hot Encoding and Standardization.
   - Balance the dataset using SMOTE (if necessary).

2. **Train the Model**:
   Run the training script to train the MLP model.
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model**:
   - The model will evaluate its performance on a test set and produce metrics such as accuracy and a confusion matrix.

4. **Visualize Results**:
   - Training accuracy and validation accuracy.
   - Confusion matrix for classification results.

## Results
### Test Accuracy
- **Accuracy:** 61.75%

### Confusion Matrix
| True Class         | Predicted Negative + Neutral | Predicted Positive |
|--------------------|-----------------------------|--------------------|
| Negative + Neutral | **211**                     | 59                 |
| Positive           | **95**                      | 35                 |

### Confusion Matrix

![Confusion Matrix](https://github.com/lishenyu1024/MoodAnalyzer/blob/725b362f398dfc64e038868ef8dd8f82803556e2/pics/matrix.jpg)

**Observations:**
- The model performs well in identifying the `Negative + Neutral` class but struggles to correctly classify the `Positive` class.
- There is significant misclassification of positive instances as `Negative + Neutral`.

### Training and Validation Performance
- **Training Accuracy:**
  - The model achieves progressively higher accuracy on the training set, nearing 90%.

![Training Accuracy](https://github.com/lishenyu1024/MoodAnalyzer/blob/725b362f398dfc64e038868ef8dd8f82803556e2/pics/Epoch.jpg)

- **Validation Accuracy:**
  - 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.6015 - loss: 1.1586 
  - Test Accuracy: 0.6275000166893005

  - Validation accuracy remains relatively stable around 60-65%, indicating potential overfitting to the training data.

## Future Improvements
- Use advanced NLP models (e.g., BERT, GPT) to capture semantic information.
- Perform hyperparameter tuning for better optimization.
- Expand dataset size and diversity to improve generalization.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Special thanks to the contributors of `scikit-learn`, `TensorFlow`, and `imblearn` for providing the tools used in this project.
