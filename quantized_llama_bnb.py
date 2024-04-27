from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-2-7b-bnb-4bit", device_map='cuda')

prompt = "My name is Anu and my favourite"
print("Prompt : ", prompt)
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')
generate_ids = model.generate(inputs.input_ids, max_length=50)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response : ", response)
