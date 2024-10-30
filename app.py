import os
import requests
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from keras.models import load_model
import cv2

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def generate_content(api_key, prompt):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    return response.json() if response.status_code == 200 else None


covid_model = load_model('models/covid19.h5')
def predict_covid(image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.reshape(img_resized, [1, 224, 224, 3])

    # Predict using the model.
    prediction_array = covid_model.predict(img_array)
    result = prediction_array.argmax(axis=-1)

    # Determine prediction label and color
    prediction = 'normal' if result[0] == 1 else 'covid'
    color = (0, 255, 0) if prediction == 'normal' else (0, 0, 255)

    # Resize for display and add prediction text
    result_img = cv2.resize(img_resized, (600, 600))
    cv2.putText(result_img, prediction, (25, 25), font, 1, color, 2, cv2.LINE_AA)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_image.jpg")
    cv2.imwrite(output_path, result_img)

    return prediction, output_path

def predict(values, dic):
    api_key = "AIzaSyBEeRHaeIg2gEo5vRHKP7xW2k33blZS2p8"  # Fetch the API key from environment variables
    text = ""  # Initialize text to an empty string

    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl', 'rb'))
        values = np.asarray(values)
        pred = model.predict(values.reshape(1, -1))[0]
        disease_name = "diabetes"
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        values = np.asarray(values)
        pred = model.predict(values.reshape(1, -1))[0]
        disease_name = "breast cancer"
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl', 'rb'))
        values = np.asarray(values)
        pred = model.predict(values.reshape(1, -1))[0]
        disease_name = "heart disease"
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl', 'rb'))
        values = np.asarray(values)
        pred = model.predict(values.reshape(1, -1))[0]
        disease_name = "kidney disease"
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl', 'rb'))
        values = np.asarray(values)
        pred = model.predict(values.reshape(1, -1))[0]
        pred = 1 - pred  # Reverse the prediction (0 becomes 1 and 1 becomes 0)
        disease_name = "liver disease"
    else:
        return None, "unknown disease", text

    # Call the API to get content
    suggestion = generate_content(api_key,
                                  (
        "Guidelines: Format your response using only HTML tags, without any markdown or other formatting styles."
        "You are an AI medical advisor. Start with a heading wrapped in <h1>, `Hi, I'm your AI advisor!`" + "The topic for this chat is" + disease_name + ", you MUST provide a very concise description of this topic."
        "The likelihood of " + disease_name + " is " + str(pred) + " (1 indicates positive, 0 indicates negative). Accordingly show your concern in TWO lines only."
        "Next, present the submitted parameters in a structured, user friendly format like `parameter name: an explanation of WHY and HOW this affects the probability of the topic negatively in one sentence. Observed value is: value` with each one being on a new line (add <Br><Br> before each parameter). The parameters are: "
        + str(dic) + 
        ". Ensure to only explain the parameters present in the submitted values. "
        "Keep each explanation concise and relevant, using only the parameters provided in the input. Add <Br> after each explanation sentence. Wrap each parameter name in <strong> tags. "
        "Conclude with 'Feel free to ask me anything else.' Maintain a consistent font size for the explanations."
    )
                                  )
    if suggestion and 'candidates' in suggestion and suggestion['candidates']:
        text = suggestion['candidates'][0]['content']['parts'][0]['text']
        print(text)  # Print to console for debugging

    return pred, disease_name, text

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')


@app.route("/covid", methods=['GET', 'POST'])
def covidPage():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(file_path)

            prediction, result_image_path = predict_covid(file_path)  # Predict COVID-19 status
            

            prompt = (
                "Guidelines: Format your response using only HTML tags, without any markdown or other formatting styles."
        "You are an AI medical advisor. Start with a heading wrapped in <h1>, `Hi, I'm your AI advisor!`" + "The topic for this chat is Covid" + 
        ", you MUST provide a very concise description of this topic."
        "You must conclude with 'Feel free to ask me anything else.' Maintain a consistent font size for the explanations. You MUST NOT say that you cannot provide medical advice. YOU ARE A MEDICAL ADVISOR. PROVIDING ADVICE IS YOUR JOB. DO NOT ASK TO CONSULT A DOCTOR. GIVE COMMON SUGGESTIONS. Remember NO MARKDOWN. YOU MUST NOT PROVIDE ANY DISCLAIMER. NO DISCLAIMERS."
        "Below is the user's test results. Accordingly show your concern in TWO lines only, or congratulate, and then comply with the user's requests."
        
    
            )


            # Prepare advice based on the prediction result
            if prediction == 'covid':
                prompt += "User has positive covid results based on X-ray analysis, give advice in the form of bullets using HTML formatting and NOT markdown."
                print(prompt)
            else:
                prompt += "User does not have covid according to X-ray analysis. Congratulate and give advice to stay safe."

            # Call the API to get the advice
            api_key = "AIzaSyBEeRHaeIg2gEo5vRHKP7xW2k33blZS2p8"
            suggestion = generate_content(api_key, prompt)
            text = suggestion['candidates'][0]['content']['parts'][0]['text'] if suggestion and 'candidates' in suggestion and suggestion['candidates'] else ""

            return render_template('imagePredict.html', prediction=prediction, image_path=result_image_path, text=text)  # Pass prediction, image path, and text to template

    return render_template('covid.html')

@app.route("/chat", methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return {"error": "No message provided"}, 400
        
        # Call your generative AI API to get a response (placeholder implementation)
        response_text = generate_content("AIzaSyBEeRHaeIg2gEo5vRHKP7xW2k33blZS2p8", user_message)  # Make sure to pass the correct prompt
        if response_text:
            return {"response": response_text['candidates'][0]['content']['parts'][0]['text']}, 200
        else:
            return {"error": "Failed to get response from AI"}, 500
            
    except Exception as e:
        return {"error": str(e)}, 500




@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred, disease_name, text = predict(to_predict_list, to_predict_dict)  # Unpacking the tuple
    except Exception as e:
        print(e)
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred, disease_name=disease_name, submitted_values=to_predict_dict, text=text)  # Passing submitted_values


if __name__ == '__main__':
    # Set default to 5000 if PORT is not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)