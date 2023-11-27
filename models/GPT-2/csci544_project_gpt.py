import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd

# Define custom dataset class for GPT-2
class SVAMPDataset(Dataset):
    def __init__(self, questions, equations, tokenizer, max_length=512):
        self.questions = questions
        self.equations = equations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        equation = self.equations[idx]

        # Concatenate question and equation for training
        text = question + " [SEP] " + equation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# Function to calculate accuracy
def calculate_accuracy(model, tokenizer, data_loader, device):
    model.eval()
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["input_ids"].to(device)  # In GPT the labels are the input_ids itself

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).float().sum().item()

    accuracy = correct_predictions / len(data_loader.dataset)
    return accuracy

# Load SVAMP dataset with only the first 2000 entries
def load_data(file_path, limit=None):
    data = pd.read_csv(file_path, skip_blank_lines=True)
    if limit:
        data = data.head(limit)  # Limit the data to the first 'limit' entries
    return data

# Prepare data with a limit
def prepare_data(data, tokenizer, batch_size=2, limit=None):
    if limit:
        data = data.head(limit)
    questions = list(data['Question'])
    equations = list(data['Equation'])
    dataset = SVAMPDataset(questions, equations, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Training loop
def train(model, tokenizer, dataloader, device, num_epochs=10):
    optimizer = AdamW(model.parameters(), lr=0.005)
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        training_accuracy = calculate_accuracy(model, tokenizer, dataloader, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Training Accuracy: {training_accuracy}")

    # Save the fine-tuned model
    model.save_pretrained("gpt2_for_svamp")
    tokenizer.save_pretrained("gpt2_for_svamp")

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set the EOS token as the PAD token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_file_path = 'C:\\Users\\malha\\OneDrive\\Desktop\\USC\\SEM_3\\CSCI_544\\Paper\\mawps-asdiv-a_svamp\\train.csv'
    test_file_path = 'C:\\Users\\malha\\OneDrive\\Desktop\\USC\\SEM_3\\CSCI_544\\Paper\\mawps-asdiv-a_svamp\\dev.csv'

    train_data = load_data(train_file_path, limit=2000)
    test_data = load_data(test_file_path)

    train_dataloader = prepare_data(train_data, tokenizer, limit=2000)
    test_dataloader = prepare_data(test_data, tokenizer)

    # Train the model
    train(model, tokenizer, train_dataloader, device)

    # Calculate test accuracy
    test_accuracy = calculate_accuracy(model, tokenizer, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy}")

