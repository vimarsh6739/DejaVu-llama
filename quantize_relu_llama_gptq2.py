from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

model_name = '/shared/vsathia2/hf_models/relu-llama'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quantizer = GPTQQuantizer(bits=8, dataset="wikitext2", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

save_folder = "/shared/vsathia2/hf_models/relu-llama-8bit-gptq"
quantizer.save(model,save_folder)
tokenizer.save_pretrained(save_folder)

