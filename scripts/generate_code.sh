
for i in `seq 0.0 0.1 1.0`; do  
    echo "called with $i" ; 
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python3 -m scripts.generate_code --temperature $i
done
