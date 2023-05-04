# Content-Based-Image-Retrieval

# **Content-Based Image Retrieval System**

This code builds a content-based image retrieval system using the ResNet50 model with transfer learning for feature extraction. The system works by extracting features from the images in a dataset of fashion products and then using pairwise distance metrics to recommend similar products based on a user's input.

The system uses several Python libraries, including Streamlit for the user interface, NumPy and pandas for data manipulation, Matplotlib and Seaborn for data visualization, TensorFlow and Keras for building and training the ResNet50 model, Scikit-learn for calculating pairwise distances, and Pillow for handling image processing.

## **Requirements**
- Python 3.6 or higher
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- Scikit-learn
- Pillow

## **Installation**
Clone the repository: git clone https://github.com/Chinnapani9439/Content-Based-Image-Retrieval.git
Install the required libraries: pip install -r requirements.txt

## **Usage**
Ensure all requirements are installed and up-to-date
Run streamlit run main.py from the project directory to launch the Streamlit app.
Use the sidebar to select the gender, subcategory, and product ID of the item you want to find similar products for.
The system will display the selected product and recommend similar products based on pairwise distances.

## **Acknowledgements**
The code was written by **Kiran Kumar** as part of a project for **Personal Project**. The dataset used in this project was collected from **Kaggle**.
