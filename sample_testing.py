import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the fine-tuned model
loaded_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_model_15_epochs")

# Tokenize new input data
new_input = "itching, nodal_skin_eruptions, dischromic _patches,"
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_input = bert_tokenizer.encode_plus(new_input, add_special_tokens=True, truncation=True, max_length=512, return_tensors='pt')

# Load the dataset from Excel
dataset_path = "C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\healthcare_chatbot_dataset.xlsx"
df = pd.read_excel(dataset_path)

# Modifying the label column accordingly
label_column = 'Disease'

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# Ensure the number of classes in label encoder matches the model
if len(label_encoder.classes_) != loaded_model.config.num_labels:
    print("Number of classes in label encoder does not match the model.")
    # Handle the mismatch, either by retraining with the correct classes or using the correct label encoder

# Make predictions
with torch.no_grad():
    output = loaded_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])

# Interpret the output (e.g., apply softmax for probabilities)
probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
predicted_label = torch.argmax(probabilities).item()

# Ensure the predicted label is within the valid range
if predicted_label >= len(label_encoder.classes_):
    print(f"Warning: Predicted label {predicted_label} is out of bounds for the given classes.")

# Map the label back to its original class using label_encoder
predicted_class = label_encoder.classes_[predicted_label]
print(f"Predicted Disease: {predicted_class}")
