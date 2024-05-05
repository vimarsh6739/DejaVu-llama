echo "relu-llama-4bit-bnb"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/relu-llama-4bit-bnb/ \
    --tasks wikitext \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/relu-llama-4bit-bnb

echo "llama-8bit-gptq"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/llama-8bit-gptq/ \
    --tasks wikitext \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/llama-8bit-gptq
