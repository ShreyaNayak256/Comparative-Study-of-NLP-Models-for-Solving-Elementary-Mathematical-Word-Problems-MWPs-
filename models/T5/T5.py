import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from tqdm import tqdm
import pandas as pd

data = pd.read_csv('train.csv', skip_blank_lines=True)

# Define your custom dataset class
class MyDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "labels": labels.squeeze()}

# Load your dataset
# Replace the following lines with loading your own dataset
texts = list(data['Question'])[:200]
targets = list(data['Equation'])[:200]

# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Create a DataLoader for your dataset
dataset = MyDataset(texts, targets, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=0.001)
num_epochs = 4

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        model.eval()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)
        correct = (predicted_ids == labels).sum().item()
        total_correct += correct
        total_samples += labels.numel()

        loss = outputs.loss
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()


    print(f"Loss {total_loss/len(dataloader)} Accuracy : {total_correct/total_samples}")

# Save the fine-tuned model
model.save_pretrained("your_fine_tuned_model")
tokenizer.save_pretrained("your_fine_tuned_model")

def calculate_accuracy(data,model):
    total_correct = 0
    for i,input_text in enumerate(data['Question']):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        model.eval()
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print(f"Predicted : {output_text}")
        # print(f"True : {data['Equation'][i]}")
        if output_text == data['Equation'][i]:
            total_correct+=1
    return total_correct/len(data)

test_data = pd.read_csv('dev.csv')
texts = list(test_data['Question'])[:100]
targets = list(test_data['Equation'])[:100]
dataset_test = MyDataset(texts, targets, tokenizer)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
calculate_accuracy(test_data, model)
calculate_accuracy(data, model)
