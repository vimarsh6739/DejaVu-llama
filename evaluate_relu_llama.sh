:'
echo "vanilla-llama"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/model/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/vanilla-llama


echo "relu-llama"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/relu-llama/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/relu-llama
'
echo "relu-llama-4bit-gptq"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/relu-llama-4bit-gptq/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/relu-llama-4bit-gptq


echo "relu-llama-8bit-gptq"
lm_eval \
    --model hf \
    --model_args pretrained=/shared/vsathia2/hf_models/relu-llama-8bit-gptq/ \
    --tasks cb,copa,lambada,openbookqa,piqa,rte,winogrande \
    --batch_size 8 \
    --device cuda:0 \
    --log_samples \
    --output_path output/relu-llama-8bit-gptq
