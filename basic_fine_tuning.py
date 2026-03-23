from datasets import load_from_disk
from transformers import BartTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BartForConditionalGeneration
import evaluate
import numpy as np
import torch

# ==================== Configuration ====================
MODEL_NAME = "facebook/bart-base"
MAX_TARGET_LENGTH = 256
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
seed = 42
# ==================== Load Model & Tokenizer ====================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Ensure pad_token_id is set correctly
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id

# ==================== Load Datasets ====================
# Load from single DatasetDict directory
processed_datasets = load_from_disk("./processed_cnn_dailymail")

# Extract splits
train_tokenized_data = processed_datasets['train']
train_tokenized_data = train_tokenized_data.shuffle(seed=seed).select(range(20))

validation_tokenized_data = processed_datasets['validation']
validation_tokenized_data = validation_tokenized_data.shuffle(seed=seed).select(range(4))

test_tokenized_data = processed_datasets['test']
test_tokenized_data = test_tokenized_data.shuffle(seed=seed).select(range(4))

# ==================== Training Arguments ====================
training_args = Seq2SeqTrainingArguments(
    output_dir="bart_cnndailymail_model",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=1,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=2,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard",
)

# ==================== ROUGE Metric ====================
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize for ROUGE (optional but recommended)
    decoded_preds = ["\n".join(pred.strip().split("\n")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split("\n")) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    
    # Convert to percentage
    result = {key: round(value * 100, 4) for key, value in result.items()}
    
    # Add mean length for reference
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

# ==================== Initialize Trainer ====================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_data,
    eval_dataset=validation_tokenized_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# ==================== Training ====================
print("Starting training...")
trainer.train()

# ==================== Save Final Model ====================
trainer.save_model("bart_cnndailymail_model_final")

# ==================== Test Set Evaluation ====================
print("Evaluating on test set...")
test_results = trainer.evaluate(test_tokenized_data)
print(f"Test Results: {test_results}")




####################################################################################
## To run tensorboard

# pip install tensorbard
# tensorboard --logdir=logs