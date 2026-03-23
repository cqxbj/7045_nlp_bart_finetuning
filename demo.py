
### DEMO 1:  Mask filling example with pre-trained model: bart-base

from transformers import AutoTokenizer, BartForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")



TXT = """
Manchester United <mask> Club, 
commonly referred to as Man United (often stylised as Man Utd) or simply United, 

"""
input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)

# Print formatted results
print("=" * 60)
print("DEMO 1: Mask Filling with BART-base Pre-trained Model")
print("=" * 60)

print(f"{'Rank':<5} | {'Token':<20} | {'Probability':<10}")
print("-" * 40)
for i, (pred, val) in enumerate(zip(predictions, values)):
    token = tokenizer.decode(pred)
    prob = val.item() * 100
    print(f"{i+1:<5} | {token:<20} | {prob:.2f}%")



### DEMO 2:  Mask filling example with pre-trained model: bart-base

### DEMO 3:  Mask filling example with pre-trained model: bart-base

### DEMO 4:  Mask filling example with pre-trained model: bart-base

### DEMO 5:  Mask filling example with pre-trained model: bart-base

### DEMO 6:  Mask filling example with pre-trained model: bart-base
