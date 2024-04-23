from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("./model")
model = LlamaForCausalLM.from_pretrained("./model")
prompt = "Once upon a time, there was thirsty crow."
print("Prompt : ", prompt)
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=150)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response : ", response)
