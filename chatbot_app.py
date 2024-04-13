
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd

app = Flask(__name__)

# Load the fine-tuned model
loaded_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_model_15_epochs")

# Load the dataset from Excel
dataset_path = "C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\healthcare_chatbot_dataset.xlsx"
df = pd.read_excel(dataset_path)

# Modifying the label column accordingly
label_column = 'Disease'

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# Initialize an empty list to store label classes
label_classes = []
if not label_classes:
    label_classes = list(label_encoder.classes_)

# Ensure label encoder is fitted with the current label classes
label_encoder.classes_ = label_classes

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    global label_classes

    user_input = request.form['symptoms']

    # Tokenize input
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_input = bert_tokenizer.encode_plus(user_input, add_special_tokens=True, truncation=True, max_length=512, return_tensors='pt')

    # Make predictions
    with torch.no_grad():
        output = loaded_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])

    # Interpret the output (e.g., apply softmax for probabilities)
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_label = torch.argmax(probabilities).item()

    # Map the predicted class back to the corresponding disease name
    predicted_class = label_encoder.classes_[predicted_label]

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
