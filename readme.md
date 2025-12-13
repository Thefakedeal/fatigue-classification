# Fatigue Classification Using Deep Learning

This repository contains the code and report for an image-based fatigue detection project using PyTorch.

## Contents
- `notebook.ipynb` – Jupyter notebook with training, testing, evaluation, metrics, and visualizations.
- `streamlit_demo.py` – Streamlit app for real-time fatigue prediction using the trained models.
- `report.pdf` – Assignment report in IEEE-style format.

## Dataset
- Kaggle Fatigue Dataset [link](https://www.kaggle.com/datasets/rihabkaci99/fatigue-dataset)
- Images are labeled as `Fatigue` and `Non-Fatigue`.

## Models
- Pre-trained: ResNet50, VGG16, MobileNetV2
- Custom CNN with 2 convolutional and 2 fully connected layers

### How to Run
1. Get the dataset from Kaggle and extract it.  
2. Update the data path in the notebook: `folder_path = "/path/to/your/dataset"`  
3. Install requirements: `pip install -r requirements.txt`  
4. Run the notebook (`notebook.ipynb`) to train and evaluate models.  
5. Run the Streamlit demo: `streamlit run streamlit_demo.py`  
Make sure the trained models and `transform.pth` are in the project folder.
