import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules="all-linear",
# )


model_id = "SparseLLM/ReluLLaMA-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#gptq_config = GPTQConfig(bits=4, dataset="wikitext2", tokenizer = tokenizer)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)
max_seq_length = 150
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda:0', quantization_config = bnb_config,torch_dtype=torch.bfloat16)
# model.resize_token_embeddings(len(tokenizer))

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset["train"],
#     peft_config=peft_config,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
# )
# trainer.train()

prompt = "My name is Delilah and my favourite"
print("Prompt : ", prompt)
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')
generate_ids = model.generate(inputs.input_ids, max_length=max_seq_length)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response : ", response)
