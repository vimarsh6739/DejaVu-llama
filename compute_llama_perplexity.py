from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("/shared/vsathia2/model/")
model = AutoModelForCausalLM.from_pretrained("/shared/vsathia2/model/", device_map='cuda')
#tokenizer = LlamaTokenizer.from_pretrained("./model")
#model = LlamaForCausalLM.from_pretrained("./model")
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
#test = load_dataset("c4", "en", split="validation")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")



prompt = "Write a story which start with the sentence - Once upon a time, there was a tiger. "
print("Prompt : ", prompt)
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')
generate_ids = model.generate(inputs.input_ids, max_length=256)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response : ", response)
max_length = model.config.max_length
stride = 512
seq_len = encodings.input_ids.size(1)
device = "cuda"

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("llama2 4-bit bnb perplexity on wikitext2 :", ppl)
