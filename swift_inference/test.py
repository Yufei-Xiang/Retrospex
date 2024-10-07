# from eval_utils import get_model_output
import csv
import pandas as pd
import os
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from eval_agent_fast_slow import parse_args, normalize
from eval_utils import load_model

# string = "task29-0-134"
# numbers = re.findall(r'\d+', string)
# print(numbers)

def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()


def get_dataset_swift():
    dataset = {"observations":[], 'taskDes':[], "freelook":[], "inv":[], "actions":[], "next_observations":[], "next_look":[], "next_inv":[], "rewards":[], "terminals":[]}
    filedir = '/home/nctu/xyf/enlighten2/LLMAgent/methods/SWIFT/fast_logs/train_swift'
    file_lst = os.listdir(filedir)
    json_file = [file for file in file_lst if os.path.splitext(file)[1]=='.json']
    trajs = 0
    print(json_file)
    for file_name in json_file:
        with open(os.path.join(filedir, file_name), 'r') as f:
            data = json.load(f)
            total_traj = len(data.keys())
            trajs += total_traj
            for i in range(total_traj):
                key = list(data.keys())[i]
                temp = data[key]
                task_des = temp["history"]["taskDescription"]
                # print(temp)
                history = temp['history']['history']
                # print(history)
                for j in range(len(history)-1):
                    dataset["taskDes"].append(clean(task_des))
                    current_step = history[j]
                    next_step = history[j+1]
                    reward = 0
                    if j == 0:
                        dataset["observations"].append("[OBSERVATION]: Start now.")
                        reward = float(current_step["score"])
                    else:
                        pre_step = history[j-1]
                        reward = float(current_step["score"])-float(pre_step["score"])
                        dataset["observations"].append(clean('[OBSERVATION]: '+pre_step["observation"]))
                    
                    terminal = 0
                    if current_step["isCompleted"]=="true":
                        terminal = 1
                    dataset["actions"].append(current_step["action"])
                    dataset["freelook"].append(clean(current_step["freelook"]))
                    dataset["inv"].append(clean(current_step["inventory"]))
                    dataset["next_observations"].append(clean('[OBSERVATION]: '+current_step["observation"]))
                    dataset["next_look"].append(clean(next_step["freelook"]))
                    dataset["next_inv"].append(clean(next_step["inventory"]))
                    dataset["rewards"].append(reward)
                    dataset["terminals"].append(terminal)

    data_pd = pd.DataFrame(dataset)
    data_pd = data_pd.sample(frac=1)
    dataset['taskDes'] = list(data_pd['taskDes'].values)
    dataset["observations"] = list(data_pd['observations'].values)
    dataset["actions"] = list(data_pd['actions'].values)
    dataset["freelook"] = list(data_pd['freelook'].values)
    dataset["inv"] = list(data_pd['inv'].values)
    dataset["next_observations"] = list(data_pd["next_observations"].values)
    dataset["next_look"] = list(data_pd["next_look"].values)
    dataset["next_inv"] = list(data_pd["next_inv"].values)
    dataset["rewards"] = list(data_pd['rewards'].values)
    dataset["terminals"] = list(data_pd['terminals'].values)
    print(trajs)
    return dataset

# colums = ['taskDes','next_look','look','inv','next_inv','score','pre_score','next_obs','observation','action','action_lst','isCompleted']
# with open("trajsWithActionSamples2.csv",'w') as f:
#     writer = csv.DictWriter(f, fieldnames=colums)
#     writer.writeheader()


def get_df():
    data = pd.read_csv("trajsWithActionSamples.csv")
    data.insert(data.shape[1], 'step', 0)
    current_s = 0
    flag = 0
    count = 0
    for index,row in data.iterrows():
        if row['score'] < current_s:
            flag = 1
            count = 0
        if flag == 1:
            data.at[index, 'step'] = 0
            current_v = row['variation']
            current_task = row['taskIdx']
            flag = 0
        else:
            count += 1
            data.at[index, 'step'] = count
    return data

# df = pd.read_csv("trajsWithActionSamples2.csv")
# print(df.size)
# print(df['look'].head(5))
# print(df['next_look'].head(5))
# print(df['observation'].head(5))
# print(df['next_obs'].head(5))
# index_lst = df.loc[(df['taskIdx']=='0')&(df['variation']==0)].index.tolist()
# print(index_lst)
# print(df['taskIdx'].unique())

# dataset = get_dataset_swift()
# print(len(dataset['inv']))
# for i in range(l):
#     if data.loc[i,'step'] == 0:
#         data.loc[i, 'observation'] = data.loc[i, 'freelook']
#         data.loc[i, 'reward'] = float(data.loc[i, 'score'])
#     else:
#         data.loc[i, 'observation'] = data.loc[i-1, 'next_observation']
#         data.loc[i, 'reward'] = float(data.loc[i, 'score'])-data.loc[i-1, 'reward']
#     if i < l-1:
#         if data.loc[i+1, 'step'] != 0:
#             data.loc[i, 'next_look'] = data.loc[i+1, 'freelook']
#             data.loc[i, 'next_inv'] = data.loc[i+1, 'freeinv']

def get_model_output(args, input_str, tokenizer, lm_model, device, logger): 
    input_ids = tokenizer(input_str, return_tensors="pt", max_length=args["max_input_len"] , truncation=True).input_ids

    sample_outputs = lm_model.generate(
        input_ids.to(device),
        max_length=16,
        num_return_sequences=args['beams'],
        num_beams=args['beams'],
    )
 
    lm_pred = sample_outputs

    # Take the first prediction that is not "look around"
    logger.info("Top N Predictions:")
    predStrs = []
    for i, pred in enumerate(lm_pred):
        text = tokenizer.decode(pred)
        # text = post_process_generation(text)
        logger.info("\t" + str(i) + "\t" + str(text) )
        predStrs.append(text)

    return predStrs

def cal_prob(target_text,input_text,model,tokenizer,device,args):
    #将input_text转换为t5模型的输入格式
    encodings = tokenizer(input_text, return_tensors="pt",max_length=args["max_input_len"] , truncation=True)
    print(encodings)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    print(encodings)
    #将target_text转换为t5模型的输出格式
    labels = tokenizer.encode(target_text, return_tensors="pt", max_length=args["max_input_len"],  truncation=True).to(device)
    #由labels生成decoder_input_ids，需要在前面补0使得长度与labels相同
    decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(device)

    #计算生成text的概率
    outputs = model(**encodings, labels=labels,decoder_input_ids=decoder_input_ids)
    loss = outputs[0]
    text_prob=torch.exp(-loss)#**(len(target_text))
    return text_prob

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lm_model, tokenizer, sbert_model, llm = load_model(args, device)
input_text = "Your task is to boil lead. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter. </s> Time: 1; Score: 0; </s> Action history: </s> <extra_id_10> look around (+0) --> N/A |  </s> Current environment: This room is called the bathroom. In it, you see:  | the agent | a substance called air | a bathtub, which is turned off. In the bathtub is: nothing. | a glass cup  | a picture | a sink, which is turned off. In the sink is: nothing. | a toilet. In the toilet is: A drain, which is open, a substance called water. | You also see: | A door to the kitchen  |  </s> Current inventory: In your inventory, you see: | an orange |  </s> Visited rooms: bathroom </s>  What action should you do next? </s>"
target_text = "go to kitchen"
prob = cal_prob(target_text, input_text, lm_model, tokenizer, device, args)
print(prob)

# lst = [1,2,3,4]
# t = torch.tensor(lst)
# normalize(t)
# print(t)
