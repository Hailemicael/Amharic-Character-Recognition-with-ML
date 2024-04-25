# Amharic Character Recognition
# Table of Contents
1. [Introduction](##introduction)
2. [Motivation](##motivation)
3. [Dataset](##dataset)
4. [Methodology](##methodology)
    - [Data Loading and Preprocessing](###data-loading-and-preprocessing)
    - [Dimensionality Reduction](###dimensionality-reduction)
        - [PCA](####PCA)
    - [Model Training](###model-training)
    - [Evaluation and Analysis](###evaluation-and-analysis)
5. [Results](##results)
6. [Usage](##usage)
7. [Future Work](##future-work)
8. [Contributors](##contributors)


## Introduction

This project aims to recognize Amharic characters using machine learning techniques. Its primary objective is to develop a model capable of accurately classifying Amharic characters from images. The project employs Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for dimensionality reduction. Additionally, it utilizes various classifiers such as Support Vector Machines (SVM), Logistic Regression, and K-nearest neighbors (KNN) for classification. The performance of each classifier is evaluated before and after applying feature extraction techniques. 

## Motivation
The motivation behind this project is to bridge the gap in existing character recognition systems that primarily focus on Latin characters. Amharic is one of the most widely spoken languages in Ethiopia, and having accurate character recognition systems can facilitate tasks such as text processing, language learning, and document digitization for Amharic speakers.

## Dataset
The dataset consists of images of handwritten Amharic characters. Each image is labeled with the corresponding character it represents. The dataset has been preprocessed to ensure consistency in image size and format, making it suitable for training machine learning models. It comprises 4200 characters, including some augmented images, with a total of 14 distinct characters for classification.

## Methodology
### Data Loading and Preprocessing
The images are loaded and preprocessed to convert them into a format suitable for training machine learning models. This includes resizing, normalization, and flattening of the image data, as well as shuffling.
```python
# Function to load images and labels
def load_images_and_labels(dataset_dir):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        for file in files:
            with Image.open(os.path.join(root, file)) as img:
                img_resized = img.resize((64, 64)).convert('L')
                img_array = np.array(img_resized).flatten()
                data.append(img_array)
                labels.append(label)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

# Normalize data
def normalize_data(data):
    data = np.array(data)
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Load and preprocess data
dataset_dir = "/home/hailemicael/ml_pro/dataset"
data, labels = load_images_and_labels(dataset_dir)
data = normalize_data(data)
```
### Dimensionality Reduction
PCA and LDA techniques are applied to reduce the dimensionality of the image data while preserving important features. This helps in improving computational efficiency and reducing overfitting.

#### PCA (Principal Component Analysis)

PCA is a statistical method used to reduce the dimensionality of data while retaining most of its variation. In this project, PCA is employed to transform the image data into a lower-dimensional space. Here's how PCA is implemented in Python:

```python
def apply_pca(data, alpha=0.95):
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    cov_matrix = np.dot(centered_data.T, centered_data) / (centered_data.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    evr = eig_values / np.sum(eig_values)
    cvr = np.cumsum(evr)
    k = np.argmax(cvr >= alpha) + 1

    print(f"Using {k} components to retain {cvr[k-1]*100:.2f}% of the variance")
    reduced_data = np.dot(centered_data, eig_vectors[:, :k])
    return reduced_data, eig_vectors, k

# Apply PCA to training data
transformed_x_train, eig_vectors_pca_train, k = apply_pca(X_train, alpha=0.95)

# Apply PCA to testing data using the eigenvectors obtained from training data
centered_x_test = X_test - np.mean(X_train, axis=0)
transformed_x_test = np.dot(centered_x_test, eig_vectors_pca_train[:, :k])
```
Visualization of PCA Analyzed Images:
```python
def pca_analyzed_images(data, eig_vectors, or_data, or_labels, or_shape=(64, 64), num_images=10):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        reconstructed_image = np.dot(data[i], eig_vectors[:, :data.shape[1]].T) + np.mean(or_data, axis=0)
        pca_analyzed_image = reconstructed_image.reshape(or_shape)

        axs[i].imshow(pca_analyzed_image, cmap='gray')
        axs[i].set_title(f"Label: {or_labels[i]}")
        axs[i].axis('off')
    plt.show()

# Visualize PCA Analyzed Images
pca_analyzed_images(transformed_x_train, eig_vectors_pca_train, X_train, y_train)
```
### Model Training
Various classifiers such as SVM, Logistic Regression, and KNN are trained using the transformed data. These classifiers are chosen for their effectiveness in handling multi-class classification tasks.

### Evaluation and Analysis
The trained models are evaluated using performance metrics such as accuracy, precision, and F1-score. Confusion matrices and classification reports are generated to analyze the model's performance on different classes of Amharic characters.

## Results
The results of the experiments demonstrate the effectiveness of the proposed approach in accurately classifying Amharic characters. The trained models achieve high accuracy and demonstrate robustness in handling variations in handwriting styles and character shapes.

## Usage
To replicate the experiments and visualize the results:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file using `pip install -r requirements.txt`.
3. Update the `dataset_dir` variable in the code with the path to your dataset directory.
4. Run the provided scripts or notebooks to load, preprocess, train, and evaluate the models.

## Future Work
Future work may involve:

- Experimenting with different feature extraction techniques.
- Exploring deep learning models for character recognition.
- Enhancing the dataset with more diverse samples and labels.

## Contributors
[Your Name]
