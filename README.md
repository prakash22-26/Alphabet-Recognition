 # 🅰️ Alphabet Recognition (A–Z) using CNN and Streamlit

This project is a deep learning-based alphabet recognition system that identifies handwritten or printed English letters (A–Z) using a convolutional neural network (CNN). A Streamlit web app is included for interactive predictions.

This project includes:
- CNN model trained on grayscale images of English alphabets
- Streamlit web interface for real-time prediction
- Image augmentation to improve model performance
- Organized dataset in folders (A–Z)

To get started:

1. Clone the repository:
   git clone https://github.com/prakash22-26/Alphabet-Recognition.git
   cd Alphabet-Recognition

2. Install required dependencies:
   pip install -r requirements.txt

To run the Streamlit web app:
   streamlit run streamlit_app.py

To retrain the model from scratch (optional):
   python model.py
Ensure your dataset is in: archive/alphabet-dataset/Train/

Input requirements:
- Upload image containing a single letter (A–Z)
- Grayscale or RGB format accepted
- Automatically resized to 64x64
- Supported formats: .jpg, .jpeg, .png

Project structure:
Alphabet-Recognition/
├── model.py                
├── streamlit_app.py       
├── model_v2.keras          
├── archive/
│   └── alphabet-dataset/   
├── requirements.txt        
└── README.md              

Dependencies:
tensorflow
opencv-python
numpy
streamlit
Pillow
scikit-learn
matplotlib

Dataset:
- A-Z folders containing training images
- Grayscale character images stored under: archive/alphabet-dataset/Train/

## 📊 Model Evaluation

The model achieved ~89% accuracy on the validation dataset.
Below is the training accuracy and loss visualization:

<img width="1503" height="715" alt="Screenshot 2025-07-23 180316" src="https://github.com/user-attachments/assets/f25c32e5-c969-45e7-a6a5-ca7d45157112" />

Here’s the training accuracy, loss, and classification report after training the CNN model:

<img width="1172" height="843" alt="Screenshot 2025-07-23 170214" src="https://github.com/user-attachments/assets/4f7e1737-792d-4c6d-85ec-78b40e02bad8" />


Author: Prakash Kumar Shah  
GitHub: https://github.com/prakash22-26
