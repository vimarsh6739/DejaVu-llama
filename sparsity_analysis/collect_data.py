import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaTokenizer
import sys
from statistics import mean
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

#keep track of current decoder layer
lid = 0
DATA_PATH = "/shared/vsathia2/sp_mlp_predictor/vanilla_llama/"
#keep track of start points(upto 50 layers)
fstarts = [0]*50
fmlp_ptrs = []
factv_ptrs = []

def actv_hook(module, inputs, outputs):
    # Need to save output of activation function
    global lid,fmlp_ptrs,factv_ptrs,fstarts
    
    #copy output to file
    if fstarts[lid] < factv_ptrs[lid].shape[0]:
        _outputs = outputs.view(-1,outputs.size(-1))
        b,e = fstarts[lid], min(fstarts[lid]+_outputs.size(0),factv_ptrs[lid].shape[0])

        #copy outputs to file
        factv_ptrs[lid][b:e] = _outputs[:(e-b)].detach().cpu().numpy()
        
        #do not update fstarts!!!

    # print(f"saving activation output at layer {lid}")
    # print(f"outputs: {outputs.shape}")
    return outputs

def mlp_hook(module, inputs, outputs):
    # Need to save input of MLP
    global lid,fmlp_ptrs,factv_ptrs,fstarts
    
    if fstarts[lid] < fmlp_ptrs[lid].shape[0]:
        
        _inputs = inputs[0].view(-1,inputs[0].size(-1))
        b,e = fstarts[lid], min(fstarts[lid]+_inputs.size(0),fmlp_ptrs[lid].shape[0])
        
        #copy contents to file
        fmlp_ptrs[lid][b:e] = (_inputs[:(e-b)].detach().cpu().numpy())
        
        #update fstarts
        fstarts[lid] += _inputs.size(0)
            
        # print(fmlp_ptrs[b:e])
        # exit()
    # print(f"saving input for mlp layer {lid}")
    # print(f"input {inputs[0].shape}")
    # print(f"")
    
    #update lid
    lid+=1

    return outputs

import re
def wikitext_detokenize(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

def prepare_data(dpath="wikitext",dname="wikitext-2-raw-v1",split="test",text_column="text"):

    data_split = load_dataset(dpath, dname, split=split)
    text_list = "\n\n".join([wikitext_detokenize(t) for t in data_split[text_column]])
    return text_list


def evaluate(model,tokenizer,device="cuda:0",dataset_path="wikitext",dataset_name="wikitext-2-raw-v1",split="train",text_column="text",n_ctx=16384,seq_len=2048):

    # prepare data
    _text = prepare_data(dataset_path,dataset_name,split,text_column)
    tokens = tokenizer(_text, truncation=False, return_tensors="pt").input_ids.to(device)
    
    # process tokens to include batch size = (n_ctx / seq_len)
     
    cutoff = 393216

    # with tqdm(range(len(tokens[0]) // n_ctx), desc="Inference data collection: - ") as progress:
    for i in tqdm(range(0,min(cutoff,len(tokens[0])),n_ctx),desc="Inference loop"):
        # Process tokens in batched manner

        start = i
        end = start + n_ctx
        
        # vectorize inner loop
        tokens_batch = tokens[:,start:end].view(-1,seq_len)
        
        token_org = tokens_batch[0,0].item()
        tokens_batch[0,0] = tokenizer.bos_token_id

        global lid 
        lid = 0

        with torch.no_grad():
            outputs = model(tokens_batch)
        # print(f"output logit shape: {outputs.logits.shape}")
        
        tokens_batch[0,0] = token_org        
    
def main(args):
    # print(args)
    global DATA_PATH
    DATA_PATH = args.data_path    
    # Init the tokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.eos_token=tokenizer.pad_token

    # Init the model itself
    # Use eager attention instead of flash/SDPA
    m = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, attn_implementation="eager")
    
    # print(m)
    #register hooks on model
    # print(len(m.model.layers))
    # exit()
    n_layers = len(m.model.layers)
    
    #initialize file pointers
    global fmlp_ptrs, factv_ptrs
    fmlp_ptrs = [] 
    for i in range(n_layers):
        fmlp_ptrs.append(np.memmap(f"{DATA_PATH}/mlp_sp_x_{i}.mmap",dtype="float16",mode="w+",shape=(393216,m.config.hidden_size)))
        factv_ptrs.append(np.memmap(f"{DATA_PATH}/mlp_label_{i}.mmap",dtype="float16",mode="w+",shape=(393216,m.config.intermediate_size)))

    with torch.no_grad(): 
        
        mlp_funcs = []
        mlp_handles = []
        actv_funcs = []
        actv_handles = []
        for i in range(n_layers):
            mlp_funcs.append(m.model.layers[i].mlp)
            actv_funcs.append(m.model.layers[i].mlp.act_fn)
            mlp_handles.append(mlp_funcs[-1].register_forward_hook(mlp_hook))
            actv_handles.append(actv_funcs[-1].register_forward_hook(actv_hook))
        # mlp_func0 = m.model.layers[0].mlp
        # mlp_handle0 = mlp_func0.register_forward_hook(mlp_hook)
        #
        # mlp_func1 = m.model.layers[1].mlp
        # mlp_handle1 = mlp_func1.register_forward_hook(mlp_hook)
        
        #begin data collection
        evaluate(m,tokenizer)
        
        # #flush changes to disk
        # for i in range(n_layers):
        #     fmlp_ptrs[i].flush()
        #     factv_ptrs[i].flush()
        #free all hooks
        for i in range(n_layers):
            mlp_handles[i].remove()
            actv_handles[i].remove()
        # mlp_handle0.remove()
        # mlp_handle1.remove()
        #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="wikitext",help="Dataset name")
    parser.add_argument("--model-path",type=str,default="/shared/vsathia2/hf_models/vanilla_llama/")
    parser.add_argument("--data-path",type=str,default="/shared/vsathia2/sp_mlp_predictor/vanilla_llama")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Select device to perform computations on")
     
    args = parser.parse_args()
    main(args)
