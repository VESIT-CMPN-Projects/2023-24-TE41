import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Import tqdm for the progress bar

# Loading our dataset
dataset_path = "C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\healthcare_chatbot_dataset.xlsx"  # Adjust the file path accordingly
df = pd.read_excel(dataset_path)

# Printing the columns of the DataFrame
print(df.columns)

# Modifying the label column accordingly
label_column = 'Disease'

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# Defining the number of labels for the BERT model
num_labels = len(df[label_column].unique())

# Loading the BERT model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
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

# Create a DataLoader for training
dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the optimizer and loss function
optimizer = Adam(bert_model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()

# Training loop with tqdm progress bar
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0

    # Wrap your DataLoader with tqdm to add a progress bar
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
        batch = tuple(t.to(device) for t in batch)
        inputs, attention_masks, labels = batch

        optimizer.zero_grad()

        outputs = bert_model(inputs, attention_mask=attention_masks, labels=labels)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save the fine-tuned model
bert_model.save_pretrained("fine_tuned_bert_model_15_epochs")
