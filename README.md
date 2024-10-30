# **Predicto: Disease Prediction App**

## **Overview**

The Disease Prediction Application is a sophisticated web platform designed to utilize machine learning models for predicting various diseases based on user-uploaded data and images. This application leverages Flask for the web framework and TensorFlow for machine learning, offering reliable predictions for conditions such as diabetes, breast cancer, heart disease, kidney disease, liver disease, and COVID-19.

## **Features**

- **Disease Prediction**: Accurately predict multiple diseases using well-trained machine learning models.
- **Image Analysis**: Upload chest X-ray images for immediate COVID-19 prediction.
- **AI Medical Advisor**: Receive tailored medical advice generated by AI.
- **User-Friendly Interface**: Intuitive design for seamless navigation and interaction.

## **Technologies Used**

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, Pickle
- **Image Processing**: OpenCV
- **APIs**: Google Generative AI for generating medical advice
- **Frontend**: HTML, CSS, JavaScript

## **Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/disease-prediction-app.git
cd disease-prediction-app
```

### **2. Set Up Your Environment**

Create a virtual environment:

```bash
conda env create -f environment.yml
conda activate disdet
```

### **3. Configure API Keys**

Add your Google API Key to the project. You can create a `.env` file or set environment variables as necessary.

### **4. Run the Application**

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## **Usage Instructions**

1. **Upload Data**: Access the disease-specific pages to input parameters or upload images.
2. **Make Predictions**: Click the "Predict" button to obtain results.
3. **View Recommendations**: Receive AI-generated advice based on the predictions.

### **Screenshots**

### **Home Page**

![Home Page](/screenshots/home_page.png)


### **Prediction Results**

| COVID-19 Prediction | Diabetes Prediction |
|---------------------|---------------|
| ![COVID-19 Prediction](/screenshots/covid.png) | ![Diabetes Prediction](/screenshots/diabetes.png) |

## **Contributing**

Contributions are welcome! To contribute to this project, please fork the repository and submit a pull request with your proposed changes.

## **Links**

- [Documentation](https://github.com/rudyoactiv/ml-diagnosis)  <!-- Replace with actual documentation link -->
- [Issue Tracker](https://github.com/yourusername/disease-prediction-app/issues)

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

For any inquiries or feedback, please contact [its.rudraneel@gmail.com].