# BERT Pre-training and Fine-tuning for Sentiment Analysis

## Project Overview

This project demonstrates the complete workflow of utilizing a BERT model, starting from pre-training on a custom text corpus and then fine-tuning it for a specific downstream task: sentiment analysis on movie reviews. The implementation is done using PyTorch and the Hugging Face Transformers library.

The core idea is to first adapt a general-purpose BERT model to a specific domain (using the `sample_text.txt` file) and then leverage that adapted knowledge to achieve high performance on a classification task (IMDB sentiment analysis).

---

## Functionality & Features

The project is split into two main phases:

### Pre-training Phase

- **Masked Language Modeling (MLM):**  
  The model learns to predict randomly masked words in a sentence, helping it understand language context and grammar.

- **Next Sentence Prediction (NSP):**  
  The model learns to predict whether two sentences are sequential or if the second sentence is a random one from the corpus. This helps the model understand sentence relationships.

- A custom `BertPretrainingDataset` class handles the data preparation, including creating sentence pairs and applying the MLM masking strategy.

- The `pretrain_bert` function executes the training loop, calculating a combined loss for both MLM and NSP tasks.

- The resulting pre-trained model is saved as `pretrained_bert_model.pth`.

### Fine-tuning Phase

- **Sentiment Analysis:**  
  The pre-trained model is adapted for a binary classification task to determine if a movie review is positive or negative.

- **Dataset:**  
  Uses the IMDB Movie Reviews dataset from Kaggle. The code includes functions to download and clean this data.

- A `TextClassificationDataset` class tokenizes and prepares the reviews and their corresponding sentiment labels.

- The `fine_tune_bert` function trains a new classification head on top of the pre-trained BERT model.

- The final fine-tuned model is saved as `fine_tuned_bert_model.pth`.

### Evaluation

- An `evaluate_model` function is provided to measure the accuracy of the fine-tuned model on a held-out validation set, demonstrating its performance on the sentiment analysis task.

---

## How the Code Works

### Key Components

#### Models

- **BertPretrainingModel:**  
  A PyTorch module that wraps a base BERT model (`prajjwal1/bert-tiny`) and adds the necessary MLM and NSP heads for the pre-training stage.

- **BertForTextClassification / BertClassificationModel:**  
  A module that adds a single linear classification layer on top of the BERT model for the fine-tuning stage.

#### Datasets

- **BertPretrainingDataset:**  
  Prepares text for MLM and NSP. It takes a list of sentences, creates pairs for NSP, and randomly masks tokens for MLM according to the original BERT paper's specifications (80% `[MASK]`, 10% random word, 10% original word).

- **TextClassificationDataset:**  
  Prepares text and labels for the classification task. It tokenizes the input reviews and converts sentiment labels (`positive`/`negative`) into numerical format (1/0).

#### Training Functions

- `pretrain_bert()`: Manages the training loop for the pre-training phase.
- `fine_tune_bert()`: Manages the training loop for the fine-tuning phase.

---

## Execution Flow

1. The `sample_text.txt` file is read and loaded into the `BertPretrainingDataset`.
2. The `BertPretrainingModel` is instantiated and trained using the `pretrain_bert` function for 10 epochs. The model's state is saved.
3. The IMDB dataset is downloaded from Kaggle, cleaned, and split into training and validation sets.
4. The pre-trained BERT model is loaded into the `BertClassificationModel`.
5. The model is then fine-tuned on the IMDB training data using the `fine_tune_bert` function for 10 epochs.
6. Finally, the `evaluate_model` function is used to calculate and print the final classification accuracy on the validation set.
