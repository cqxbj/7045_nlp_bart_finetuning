from transformers import BartTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BartForConditionalGeneration

MODEL_NAME = "facebook/bart-base"

# Initialize tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

# Initialize model
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Print configuration structure
print("\n--- Model Structure ---")
print(model.config)

# Calculate and print model size (parameter count)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# Calculate trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params}")

# Print special tokens
print("\n--- Special Tokens ---")
print(f"Vocabulary Size: {tokenizer.vocab_size}")
print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"UNK Token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
print(f"MASK Token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
print(f"All Special Tokens: {tokenizer.all_special_tokens}")