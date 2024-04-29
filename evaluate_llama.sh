echo "llama-4bit-bnb"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/llama-4bit-bnb/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/llama-4bit-bnb

echo "llama-4bit-gptq"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/llama-4bit-gptq/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/llama-4bit-gptq

echo "llama-8bit-bnb"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/llama-8bit-bnb/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/llama-8bit-bnb

echo "llama-8bit-gptq"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/llama-8bit-gptq/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/llama-8bit-gptq
