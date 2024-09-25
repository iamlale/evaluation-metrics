# Text Generation Evaluation Pipeline

This project demonstrates how to evaluate a text generation model using Hugging Face’s `transformers`, `datasets`, and `evaluate` libraries. It utilizes BERTScore, BLEU, and ROUGE metrics to evaluate the performance of a language model on a test dataset and saves the results in an Excel file.

**Prerequisites**
Before running the code, ensure that the required packages are installed. You can install them using the following command:
```
!pip install transformers datasets evaluate bert_score sacrebleu rouge_score numpy pandas
```
Additionally, if you are using Google Colab, you need to mount Google Drive to save the results.

# Code Overview
**1. Model and Tokenizer Setup**
This part loads a pre-trained causal language model and tokenizer from Hugging Face. You can replace the `model_name` and `dataset_name` variables with your custom model and dataset.
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "user/model_name"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
