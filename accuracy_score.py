import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model
fine_tuned_model_path = "fine_tuned_bert_model_15_epochs"
fine_tuned_bert_model = BertForSequenceClassification.from_pretrained(fine_tuned_model_path)
fine_tuned_bert_model.eval()

# Load the dataset
dataset_path = "C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\healthcare_chatbot_dataset.xlsx"  # Adjust the file path accordingly
df = pd.read_excel(dataset_path)

# Modifying the label column accordingly
label_column = 'Disease'

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# Loading the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing the input data
tokenized_data = df['Symptoms'].apply(
    lambda x: bert_tokenizer.encode_plus(x, add_special_tokens=True, truncation=True, max_length=512, return_tensors='pt')
)

# Find the maximum sequence length in the batch
max_len = max(tokenized_data.apply(lambda x: x['input_ids'].size(1)))

# Pad the sequences dynamically
input_ids = torch.cat([
    torch.nn.functional.pad(item['input_ids'], (0, max_len - item['input_ids'].size(1)))
    for item in tokenized_data
])

attention_masks = torch.cat([
    torch.nn.functional.pad(item['attention_mask'], (0, max_len - item['attention_mask'].size(1)))
    for item in tokenized_data
])

labels = torch.tensor(df[label_column].values, dtype=torch.long)

# Create a DataLoader for validation/test
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Make predictions on the validation/test set with progress bar
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        batch = tuple(t.to(device) for t in batch)
        inputs, attention_masks, labels = batch

        outputs = fine_tuned_bert_model(inputs, attention_mask=attention_masks)
        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Accuracy on the dataset: {accuracy * 100:.2f}%")
