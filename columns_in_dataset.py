import pandas as pd

# Load your dataset
file_path = 'C:\\Users\\Kailash\\Desktop\\Medchat a Healthcare chatbot\\dataset.xlsx'
df = pd.read_excel(file_path)

# Display the columns
columns = df.columns
print("Columns in the dataset:")
for col in columns:
    print(col)
