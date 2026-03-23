from datasets import load_from_disk
from transformers import BartTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BartForConditionalGeneration
import evaluate
import numpy as np
import torch
import os
import logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ==================== logs ====================
# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 确保 logs 目录存在
# os.makedirs("./logs", exist_ok=True)

# 设置 TensorBoard 日志目录
os.environ['TENSORBOARD_LOGGING_DIR'] = "./logs"

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

# ==================== LoRA Configuration ====================
# LoRA 超参数配置
LORA_R = 4  # LoRA 秩，通常为 4, 8, 16
LORA_ALPHA = 8  # LoRA 缩放参数
LORA_DROPOUT = 0.1  # LoRA dropout 率
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]  # BART 的目标模块

# 配置 LoRA
lora_config = LoraConfig(
    r=LORA_R,  # 秩
    lora_alpha=LORA_ALPHA,  # 缩放参数
    target_modules=LORA_TARGET_MODULES,  # 要应用 LoRA 的模块
    lora_dropout=LORA_DROPOUT,  # dropout
    bias="none",  # 是否训练偏置
    task_type=TaskType.SEQ_2_SEQ_LM,  # 任务类型：序列到序列语言模型
)

# ==================== Load Model & Tokenizer ====================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
base_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 应用 LoRA适配器到模型
lora_model = get_peft_model(base_model, lora_config)
# 打印可训练参数信息
lora_model.print_trainable_parameters()

# Ensure pad_token_id is set correctly
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.decoder_start_token_id = tokenizer.bos_token_id
lora_model.config.pad_token_id = tokenizer.pad_token_id
lora_model.config.decoder_start_token_id = tokenizer.bos_token_id

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
def training_args(logging_dir="./logs_base"):


    return Seq2SeqTrainingArguments(
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
        logging_dir=logging_dir,
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
os.makedirs("./logs_base", exist_ok=True)
os.makedirs("./logs_lora", exist_ok=True)
#LoRA
trainer_lora = Seq2SeqTrainer(
    model=lora_model,
    args=training_args(logging_dir="./logs_lora"),
    train_dataset=train_tokenized_data,
    eval_dataset=validation_tokenized_data,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
#base
trainer_base = Seq2SeqTrainer(
    model=base_model,
    args=training_args(logging_dir="./logs_base"),
    train_dataset=train_tokenized_data,
    eval_dataset=validation_tokenized_data,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)


# ==================== Training ====================
print("Starting training base model...")
trainer_base.train()
print("Starting training LoRA model...")
trainer_lora.train()

# ==================== Save Final Model ====================
#trainer.save_model("bart_cnndailymail_model_final")


# 保存 LoRA 适配器权重（轻量级）
lora_model.save_pretrained("bart_cnndailymail_model_lora")
tokenizer.save_pretrained("bart_cnndailymail_model_lora")

print(f"LoRA adapter saved to: bart_cnndailymail_model_lora")
print(f"Trainable parameters saved (much smaller than full model)")

# ==================== 推理测试 ====================
# 加载基础模型
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, "bart_cnndailymail_model_lora")

# 推理
text = """Climate change is causing more frequent and intense wildfires in California. 
Scientists say rising temperatures and prolonged drought have created conditions 
that make wildfires more likely. The 2020 wildfire season was the worst on record, 
with over 4 million acres burned. State officials are investing in forest management 
and fire prevention programs to reduce the risk. Residents are being encouraged to 
create defensible space around their homes and have evacuation plans ready."""

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=256)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)


####################################################################################
## To run tensorboard
# pip install tensorboard
# tensorboard --logdir=logs