import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Loading dataset
dataset_path = "C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\healthcare_chatbot_dataset.xlsx"
df = pd.read_excel(dataset_path)

# Define label column
label_column = 'Disease'

# Encoding labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# Tokenize input data
tokenized_data = df['Symptoms'].apply(lambda x: bert_tokenizer.encode_plus(x, add_special_tokens=True, truncation=True, max_length=512, return_tensors='pt'))

# Pad sequences
max_len = max(tokenized_data.apply(lambda x: x['input_ids'].size(1)))
input_ids = torch.cat([torch.nn.functional.pad(item['input_ids'], (0, max_len - item['input_ids'].size(1))) for item in tokenized_data])
attention_masks = torch.cat([torch.nn.functional.pad(item['attention_mask'], (0, max_len - item['attention_mask'].size(1))) for item in tokenized_data])
labels = torch.tensor(df[label_column].values, dtype=torch.long)

# Split data into train and test sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Define optimizer and loss function
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

num_epochs = 5  # You might need to increase this
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = bert_model(inputs, attention_mask=masks, labels=labels)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    bert_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Validation", leave=False):
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch
            outputs = bert_model(inputs, attention_mask=masks)
            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy on dataset: {accuracy:.4f}")

    # Early stopping condition
    if accuracy >= 0.95:
        print("Accuracy on the dataset")
        break

# Save the fine-tuned model
bert_model.save_pretrained("fine_tuned_bert_model_chatbot")

