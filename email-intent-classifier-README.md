# Email Intent Classification with DistilBERT

CS470 project: fine-tune DistilBERT to classify emails into 5 intents:
Request, Meeting, Follow-up, Complaint, and Spam.

## What it does
- PyTorch dataset + tokenization pipeline (DistilBERT tokenizer, max length 128)
- Fine-tuned `distilbert-base-uncased` for sequence classification
- TF-IDF + Logistic Regression baseline for comparison
- Inference function returning the predicted intent with per-class confidence scores
- Saved model and tokenizer for reuse

## Data
- 1,250 emails, stratified 70/15/15 split:
  - Train: 875
  - Validation: 187
  - Test: 188
- Note: the original course dataset file (`email_intents.csv`, from Google Drive in the notebook) is NOT included in this repo yet. Add it, or a sample, before claiming the notebook runs end-to-end from this repo alone.

## Training
- Batch size: 16
- Epochs: 3, with early stopping (patience 2)
- Optimizer: AdamW, learning rate 2e-5, warmup 100 steps
- Device in the original run: CPU

## Results (held-out course test set)
- Best validation accuracy: 99.47%
- Test accuracy: 100% on 188 emails, loss 0.0991
- Context: this was a clean course dataset, so treat 100% as a course-test result, not real-world inbox performance.

## Files
- `main-3.ipynb` — original Colab notebook (Drive mount + dataset path will need updating for local/GitHub use)
