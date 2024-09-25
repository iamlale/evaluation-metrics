# Text Generation Evaluation Pipeline

This project demonstrates how to evaluate a text generation model using Hugging Face’s `transformers`, `datasets`, and `evaluate` libraries. It utilizes BERTScore, BLEU, and ROUGE metrics to evaluate the performance of a language model on a test dataset and saves the results in an Excel file.

**Prerequisites**
Before running the code, ensure that the required packages are installed. You can install them using the following command:
```python
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
**2. Dataset Loading and Preparation**
The dataset is loaded from Hugging Face’s `datasets` library. The code assumes that the dataset has a conversation structure, where conversations are separated by the participant (either 'human' or 'gpt').
```python
dataset_name = "user/dataset"
dataset = load_dataset(dataset_name)

input_texts = [conv['value'] for data in dataset['test'] for conv in data['conversations'] if conv['from'] == 'human']
references = [[conv['value']] for data in dataset['test'] for conv in data['conversations'] if conv['from'] == 'gpt']
```

**3. Generating Predictions**
The `generate_predictions()` function takes the human inputs, generates responses using the model, and returns the predicted responses.
```python
def generate_predictions(model, tokenizer, input_texts, max_length=256):
    # Generates predictions for each human prompt
    ...
```
**4. Evaluation Metrics**
The code evaluates the predictions using BERTScore, BLEU, and ROUGE metrics from the `evaluate` library.

- BERTScore: A metric that evaluates the similarity between generated and reference texts using BERT embeddings.
- BLEU Score: A precision-based metric that evaluates text similarity.
- ROUGE-1, ROUGE-2, ROUGE-L: Recall-based metrics for evaluating n-gram overlaps.
```python
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

bertscore_results = bertscore.compute(predictions=predictions, references=[ref[0] for ref in references], lang="en")
bleu_results = bleu.compute(predictions=predictions, references=references)
rouge_results = rouge.compute(predictions=predictions, references=[ref[0] for ref in references])
```
**5. Saving Results to Excel**
After computing the predictions and evaluation metrics, the code saves the results (human prompts, generated responses, and reference responses) into an Excel file on Google Drive.
```python
import pandas as pd
import os

df = pd.DataFrame({
    'Human Prompt': input_texts,
    'Generated Response': predictions,
    'Reference Response': references
})

file_path = os.path.join(path, 'excel_file.xlsx')
df.to_excel(file_path, index=False)
```
**6. Running on Google Colab**
If running on Google Colab, the code mounts Google Drive and saves the results to the specified folder:
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Output Metrics**
- BERTScore - Precision, Recall, F1
- BLEU Score
- ROUGE-1, ROUGE-2, ROUGE-L
These metrics provide an evaluation of how well the generated text matches the reference responses.

**Example Results:**
```bash
BERTScore - Precision: 0.8974, Recall: 0.9021, F1: 0.8997
BLEU Score: 45.67
ROUGE-1: 0.73
ROUGE-2: 0.62
ROUGE-L: 0.70
```
**How to Run**
1. Clone the repository and open the notebook in Google Colab.
2. Install the required dependencies using `pip install`.
3. Load your custom model and dataset by modifying `model_name` and `dataset_name`.
4. Run the notebook to generate predictions and evaluate the results.
5. The final results will be saved in an Excel file on your Google Drive.

**Customization**
You can replace `model_name` with the name of any Hugging Face model you want to evaluate.
Modify `dataset_name` to use your own dataset.
Adjust the maximum token length in the `generate_predictions()` function by changing the `max_length` parameter.
