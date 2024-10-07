num_gpus=1  
seed=42
split="test_mini"
gpt_version="gpt-3.5-turbo"
# gpt_version="gpt-4"
# gpu=0

output_path="fast_logs/${split}_swift8000_0.97_0.6_refined_b_addlast_qv/"
mkdir -p $output_path
# echo "---> $output_path" 
 
iql_path="/home/nctu/xyf/enlighten2/LLMAgent/methods/SWIFT/IQLmodel/final_iql_latest_balanced_addlast.pt"
# TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES=0

# task_num="0,4,9,22"
# python eval_agent_fast_slow.py \
#     --task_nums $task_num \
#     --set ${split} \
#     --output_path $output_path\
#     --seed $seed \
#     --debug_var -1 \
#     --gpt_version $gpt_version \
#     --discount_prob 0.97 \
#     --limit_prob 0.5 \
#     --iql_path $iql_path


# if [ $num_gpus -eq 1 ]; then
#     task_nums=("0,24,19,14,27,10,15,21,20,6,9,16,12,25,22" "1,4,2,8,7,11,18,26,3,23,29,13,5,17,28")
#     L=2
# elif [ $num_gpus -eq 4 ]; then
#     task_nums=("0,12,20,16" "26,13,2,28" "22,17,3,10" "1,4,5,29" "18,14,11,15" "25,6,27,24" "19,8,9" "21,23,7")
#     L=8
# fi 

# task_nums=("0,4" "9,22")
# L=2

task_nums=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
# L=1
# task_nums=("0,1,9,10,11,14,15,16,21,22,23,24")
L=1

# output_path="fast_logs/${split}_swift_only_0/"
# # mkdir -p $output_path
# # echo "---> $output_path" 

for ((i=0; i<L; i++)); do
    task_num=${task_nums[$i]}
    ((gpu=i%num_gpus)) # the number of gpus
    echo $task_num "on" $gpu    
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_slow.py \
        --task_nums $task_num \
        --set ${split} \
        --seed $seed \
        --debug_var -1 \
        --gpt_version $gpt_version \
        --discount_prob 0.97 \
        --limit_prob 0.6 \
        --iql_path $iql_path \
        --output_path $output_path & # > /dev/null 2>&1 &
    sleep 10
done

#0.97,0.99,0.95