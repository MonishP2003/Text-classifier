import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the JSON file with the list of features
with open(r"C:\Users\monis\PycharmProjects\pythonProject\D045\Map.json") as f:
    features = json.load(f)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the configuration for the model
config = BertForSequenceClassification.config_class.from_pretrained('bert-base-uncased')
config.num_labels = len(features)

# Initialize the model with the modified configuration
model = BertForSequenceClassification(config)


# Function to classify a text based on the features
def classify_text(text):
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Convert tokens to tensor
    input_ids = torch.tensor(tokens).unsqueeze(0)

    # Perform forward pass through the model
    outputs = model(input_ids)
    logits = outputs.logits

    # Get the predicted feature index
    predicted_index = torch.argmax(logits, dim=1).item()

    # Get the predicted feature name
    predicted_feature = features[predicted_index]

    return predicted_feature


# Read text from a file and classify them
def classify_text_from_file(file_path):
    predictions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                predicted_feature = classify_text(line)
                prediction = {
                    "Text": line,
                    "Predicted Feature": predicted_feature
                }
                predictions.append(prediction)

    # Write predictions to a JSON file
    output_file_path = r"C:\Users\monis\PycharmProjects\pythonProject\D045\Map1.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(predictions, output_file, indent=4)


# Provide the path to your text file
text_file_path = r'C:\Users\monis\PycharmProjects\pythonProject\D045\Input_texts\input_2.txt'
classify_text_from_file(text_file_path)
