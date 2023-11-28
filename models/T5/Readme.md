# Fine-Tuning T5 for Equation Generation

This repository contains code for fine-tuning the T5 model on a custom dataset for the task of equation generation. The model is trained using the `transformers` library by Hugging Face.

## Getting Started

### Prerequisites

Make sure you have the required libraries installed. You can install them using the following command:

```bash
pip install pandas numpy datasets torch tqdm transformers
```

### Dataset

1. Download the training dataset (`train.csv`) and testing dataset (`dev.csv`).
2. Update the file paths in the script accordingly.

### Fine-Tuning

Run the provided script to fine-tune the T5 model on the equation generation task:

```bash
python fine_tune_equation_generation.py
```

This script loads the training data, tokenizes it using the T5 tokenizer, creates a custom dataset, and then fine-tunes the T5 model using the AdamW optimizer. The training loop runs for a specified number of epochs, and the model is saved after training.

### Evaluation

To evaluate the fine-tuned model on the testing dataset, run:

```bash
python evaluate_equation_generation.py
```

This script calculates the accuracy of the model on the testing dataset by comparing the generated equations with the ground truth.

## Results

The fine-tuned model achieves an accuracy of X% on the testing dataset.

## Model Inference

To use the trained model for inference on new data, you can utilize the `calculate_accuracy` function in the script. Replace the file path with your own data, and the function will return the accuracy of the model on the provided dataset.

```python
# Example usage
test_data = pd.read_csv('your_test_data.csv')
accuracy = calculate_accuracy(test_data, model)
print(f"Model Accuracy on Test Data: {accuracy}")
```

## Save Model

The fine-tuned model and tokenizer are saved in the "your_fine_tuned_model" directory. You can use these files for future inference or share them with others.

---

Feel free to customize the script according to your specific requirements. If you encounter any issues or have questions, please create an issue in this repository.
