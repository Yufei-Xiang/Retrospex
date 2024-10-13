import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE
import torch
import gc
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import openai
from inference import get_llama3_7b_response, get_llama3_7b_response_multi, cal_prob, MappingValidActionList, get_actions_origin
import json
import csv
from IQL.src.iql import ImplicitQLearning_webshop
import time

PYTORCH_CUDA_ALLOC_CONF=True
# CUDA_LAUNCH_BLOCKING=1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

max_memory_mapping = {0: "38GB", 1:"38GB"}

prompt1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

'''WebShop [SEP] Instruction: [SEP] Find me slim fit, machine wash women's jumpsuits, rompers & overalls with short sleeve, high waist, polyester spandex for
daily wear with color: green stripe, and size: large, and price lower than 50.00 dollars [SEP] Search'''

'''Instruction: [SEP] Find me slim fit, machine wash women's jumpsuits, rompers & overalls with short sleeve, high waist, polyester spandex for daily wear 
with color: green stripe, and size: large, and price lower than 50.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] 
B09NDS8F4V [SEP] AODONG Onesie Pajamas for Women Onesies Romper Pajamas Printed Bodycon Jumpsuit Shorts Sexy One Piece Pjs Overall [SEP] $2.99 to $7.99 
[SEP] B09PVNLVRW [SEP] Women's V-Neck Rompers Printed Jumpsuit Long Sleeve Homewear Butt Flap Pajamas One-Piece Onesies Nightwear Sexy Bodysuit [SEP] 
$17.4 to $28.67 [SEP] B099WX3CV5 [SEP] Women Aesthetic Short Sleeve Jumpsuit Bodycon Sexy V Neck Button Shorts Rompers Knitted One Piece Bodysuit Overall 
[SEP] $13.99 to $24.89 [SEP] B09Q37JQZ6 [SEP] Women's Sexy Swimsuit One Piece High Neck Halter Bikini Floral Stiching See Through Monokini Tummy Control 
Beachwear [SEP] $10.99 to $18.99 [SEP] B09P5CRVQ6 [SEP] Womens Tops Casual, Sweatshirt for Women Graphic, Womens Sweatshirt Hoodie, Women Love Print 
Sweatshirt Long Sleeve Cute Heart Sweater Pullover Tops Blouse [SEP] $22.78 [SEP] B09RV4TXKJ [SEP] ORT Bodycon Lingeries for Women,2 Piece 2 pc Sleepwear 
for Women,Sexy Printed Ugly Valentine Nightdress Loungewear Lingeries Rompers Clubwear [SEP] $13.08 [SEP] B09QGK5XHZ [SEP] WENKOMG1 Men's Long Sleeve 
Undershirt with Mask Turtleneck Hooded T-Shirt Solid Color Workout Tops Zipper Side Slit Shirts Slim Fit Sweatshirt Spring/Summer Tee Shirts(Gray,) [SEP] 
$8.39 [SEP] B09NPML43M [SEP] Womens Short Sleeve Tops, Womens Casual Dandelion Printing T-Shirts Loose O-Neck Blouse Tops Funny Graphic Tee Shirts [SEP] 
$1.01 to $1.74 [SEP] B07MGB73NJ [SEP] Viracy Women's Short Sleeve V-Neck Casual Flowy Tunic Shirt (M-3XL) [SEP] $16.99 to $26.99 [SEP] B095SX6366 [SEP] 
WYTong Women Dress Summer V Neck Swing Dress Fashion Solid Color Tie the knot Short sleeves Leisure Dress [SEP] $9.99 to $11.99'''

def initial_messages_chat():
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant!"
            },
            {
                "role": "user",
                "content": """You are web shopping.\n
                I will give you instructions about what to do.\n
                You have to follow the instructions.\n
                Every round I will give you an observation, you have to respond an action based on the state and instruction.\n
                You can use search action if search is available.\n
                You can click one of the buttons in clickables.\n
                An action should be of the following structure:\n
                search[keywords]\n
                click[value]\n
                If the action is not valid, perform nothing.\n
                Keywords in search are up to you, but the value in click must be a value in the list of available actions.\n
                Remember that your keywords in search should be carefully designed. \n
                When there exists[Back to search], it means the search action is not valid. Once you need to search different keywords, you need to click[Back to Search] first.\n
                Your response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]\n
                Attention that the goods introduction after search is not the full information and cannot exactly match the requirements, you need to click the goods and get more detail information.
                """
            },
            {
                "role": "assistant",
                "content": "Ok."
            },
            {
                "role": "user",
                "content": "First I will give you an example. Please learn from the example. Then I will give you a new task."
            },
            {
                "role": "assistant",
                "content": "Ok."
            },
            {
                "role": "user",
                "content": "Webshop \nInstruction:\n i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars \n[Search]"
            },
            {
                "role": "assistant",
                "content": "Action:search[3 ounce bright citrus deodorant sensitive skin]"
            },
            {
                "role": "user",
                "content": "Observation: \n[Back to Search] \nPage 1 (Total results: 50) \n[Next >] \n[B078GWRC1J] \nBright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce \n$10.99 \n[B078GTKVXY] \nGinger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce \n$10.99 \n[B08KBVJ4XN] \nBarrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) \n$15.95  "
            },
            {
                "role": "assistant",
                "content": "Action:think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]"
            },
            {
                "role": "user",
                "content": "Observation: OK."
            },
            {
                "role": "assistant",
                "content": "Action:click[B078GWRC1J]"
            },
            {
                "role": "user",
                "content": "Observation: \n[Back to Search] \n[< Prev] \nscent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]\nsize [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]\nBright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce \nPrice: $10.99 \nRating: N.A. \n[Description] \n[Features] \n[Reviews] \n[Buy Now] "
            },
            {
                "role": "assistant",
                "content": "Action:think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]"
            },
            {
                "role": "user",
                "content": "Observation: OK."
            },
            {
                "role": "assistant",
                "content": "Action:click[bright citrus]"
            },
            {
                "role": "user",
                "content": "Observation: You have clicked bright citrus."
            },
            {
                "role": "assistant",
                "content": "Action:click[3 ounce (pack of 1)]"
            },
            {
                "role": "user",
                "content": "Observation: You have clicked 3 ounce (pack of 1). "
            },
            {
                "role": "assistant",
                "content": "Action:click[Buy Now]"
            },
            {
                "role": "user",
                "content": "Now I will give you the new task. Please learn from the example and give solution."
            },
            {
                "role": "assistant",
                "content": "Ok."
            },
        ]
    return messages

def init_message_llama3_7B(): # 改成对应任务的prompt
    messages = [
            {
                "role": "system",
                "content": "You are a helpful, respectful and honest assistant."
            },
            {
                "role": "user",
                "content": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]"
            },
            {
                "role": "assistant",
                "content": "Ok."
            },
        ]
    return messages

def load_model_iql(path, args):
    iql = ImplicitQLearning_webshop(
        args,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args['learning_rate'])
    )
    iql.load_state_dict(torch.load(path))
    return iql

def get_available_actions(a):
    if a['has_search_bar'] == True:
        return []
    else:
        buttom_lst = a['clickables']
        action_lst = []
        for i in range(len(buttom_lst)):
            action = 'click['+str(buttom_lst[i])+']'
            action_lst.append(action)
        return action_lst

def normalize(lst):
    max_v = max(lst)
    min_v = min(lst)
    if max_v != min_v:
        result = [(i-min_v)/(max_v-min_v) for i in lst]
    else:
        result = [0.5 for i in range(len(lst))]
    return result

def decide_action(probs, q_values, step, args):
    # normalize 2 list
    # probs = torch.exp(probs)
    # probs = list(probs)
    assert len(probs) == len(q_values)
    nor_prob = normalize(probs)
    nor_q = normalize(q_values)
    print(nor_prob)
    print(nor_q)
    lam = args['discount_prob']**step
    lam = max(args['limit_prob'], lam)
    # lam = args['limit_prob']
    score = [nor_prob[i]*lam+nor_q[i]*(1-lam) for i in range(len(nor_q))]
    max_score = max(score)
    return score.index(max_score)


def filter_obs(observation):
    obs = ""
    if "WebShop" in observation[:10]:
        obs = '[SEP]'.join(observation.split('[SEP]')[3:])
    else:
        obs = '[SEP]'.join(observation.split('[SEP]')[2:])
    return obs

def save_step(row):
    colums = ['taskDes','observation','next_observation','action','reward','terminal']
    with open("trajsOnWebshop.csv",'a',newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colums)
        writer.writerow(row)

def llm_tgi(prompt, tokenizer, model):
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # print(inputs.input_ids.shape)
    # y = inputs.input_ids.repeat(5,1)
    # print(y.shape)
    #
    # Generate
    generate_input = {
        "input_ids": inputs.input_ids,
        "do_sample": False,
        "top_k": 50,
        "top_p": 0.90,
        "temperature": 0.5,
        "num_beams": 1,
        "repetition_penalty": 1.3,
    }
    generate_ids = model.generate(**generate_input, max_length=4096).to(device)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)
    l = len(prompt)
    response = output[0][l:].split('[INST]')[0].split('<|end_of_turn|>')[0].strip()
    return response

def get_prompt(conv: Conversation) -> str:
    if conv.name == 'openchat':
        ret = ''
        for role, message in conv.messages:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt()

def evaluate_web_human(env, args):
    for r in range(args['round']):
        env.reset()
        observation = env.observation
        pre_obs = str(observation)
        step = 0
        # row = {'obs':observation,'action':"Null",'action_lst':"Null",'score':0}
        while step < args['env_step_limit']:
            print(observation)
            available_actions = env.get_available_actions()
            print(type(available_actions))
            print('Available actions:', available_actions)
            action = input("please enter the action:")
            observation, reward, done, info = env.step(action)
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
            pre_obs = observation
            step += 1

    env.close()

def evaluate_web_chatmodel(env, client, args):
    for r in range(args['round']):
        env.reset()
        observation = env.observation
        step = 0
        messages = initial_messages_chat()
        messages.append({"role":"user", "content":str(observation).replace('[SEP]','\n')})
        while step < args['env_step_limit']:
            print(observation)
            chat_completion = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            response = chat_completion.choices[0].message.content
            print(response)
            messages.append({"role":"assistant", "content":response})
            action = response.replace("\n",'').split('Action:')[-1]
            if 'think' in action[:10]:
                observation = "Observation: OK."
            else:
                observation, reward, done, info = env.step(action)
                if observation[:11] == 'Instruction':
                    change_obs = observation.split('[SEP]')[2:]
                    observation = "Observation: "+("\n").join(change_obs)
                
                print(f'Taking action "{escape(action)}" -> Reward = {reward}')
                if done:
                    break
            messages.append({"role":"user", "content":str(observation).replace('[SEP]','\n')})
            step += 1

    env.close()

def evaluate_web_llama3finetuned(env, model, tokenizer, args):
    print('Yeah!')
    final_reward = 0
    for r in range(1):
        env.reset()
        observation = env.observation
        task_des = ' '.join(observation.split('[SEP]')[:3])
        step = 0
        messages = init_message_llama3_7B()
        messages.append({"role":"user", "content":str(observation)})
        done = False
        while step < args['env_step_limit']:
            # print(observation)
            response = get_llama3_7b_response(model, tokenizer, messages)
            # print(response)
            messages.append({"role":"assistant", "content":response})
            action = response.replace("\n",'').split('Action:')[-1]
            if 'think' in action[:10]:
                observation = "Observation: OK."
            else:
                # row = {'taskDes':task_des,'observation':filter_obs(observation),'action':action}
                observation, reward, done, info = env.step(action)
                # row['next_observation'] = filter_obs(observation)
                # row['reward'] = reward-final_reward
                # row['terminal'] = 0 if not done else 1
                final_reward = reward
                print(f'Taking action "{escape(action)}" -> Reward = {reward}')
                # save_step(row)
                if done:
                    break
            messages.append({"role":"user", "content":str(observation)})
            step += 1
        if not done:
            final_action = 'click[buy now]'
            if final_action in get_available_actions(env.get_available_actions()):
                observation, reward, done, info = env.step('click[buy now]')    
                final_reward = reward
    with open(args['output_file'],'a') as f:
        f.write(str(final_reward))
        f.write("\n")
    
    # with open("test_result.txt",'a') as f:
    #     f.write(str(final_reward))
    #     f.write('\n')
    
    gc.collect()
    torch.cuda.empty_cache()
    env.close()

def evaluate_web_llama3finetuned_addIQL(env, model, tokenizer, IQLmodel, sbert_model, args):
    reward_total = []
    for r in range(1):
        env.reset()
        observation = env.observation
        task_des = ' '.join(observation.split('[SEP]')[:3])
        step = 0
        messages = init_message_llama3_7B()
        messages.append({"role":"user", "content":str(observation)})
        final_reward = 0
        done = False
        action = None
        while step < args['env_step_limit']:
            # print(observation)
            responses = get_llama3_7b_response_multi(model, tokenizer, messages, args['beams'])
            # print(responses)
            avaliable_actions = get_available_actions(env.get_available_actions())
            mapped_actions = MappingValidActionList(responses, avaliable_actions, sbert_model, args['beams'])
            # mapped_actions = get_actions_origin(responses)
            # print(mapped_actions)
            if len(mapped_actions)>1:
                q_values = IQLmodel.get_q(task_des, filter_obs(observation), mapped_actions)
                # print(q_values)
                prob_scores = []
                # print(len(generate_action_lst))
                for a in mapped_actions:
                    p = cal_prob(a, messages, model, tokenizer)
                    # p = 0
                    prob_scores.append(float(p))
                action = mapped_actions[decide_action(prob_scores, q_values, step, args)]
            else:
                action = mapped_actions[0]
            observation, reward, done, info = env.step(action)
            
            final_reward = reward
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            messages.append({"role":"assistant", "content":"Action:\n"+action})
            if done:
                break
            messages.append({"role":"user", "content":str(observation)})
            step += 1
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
        if not done:
            final_action = 'click[buy now]'
            if final_action in get_available_actions(env.get_available_actions()):
                observation, reward, done, info = env.step('click[buy now]')    
                final_reward = reward
        with open(args['output_file'],'a') as f:
            f.write(str(final_reward))
            f.write("\n")
        reward_total.append(final_reward)
    # with open("test_result_iql.txt",'a') as f:
    #     f.write('\n')
    #     f.write(str(reward_total))
    #     f.write('\n')
    #     f.write("Average reward on %d random tasks is %.2f"%(args['round'],sum(reward_total)/len(reward_total)))
    #     f.write('\n\n')
    # print("Average reward on %d random tasks is %.2f"%(args['round'],sum(reward_total)/len(reward_total)))
        
    env.close()

def evaluate_web(env, model, tokenizer, args):
    for r in range(args['round']):
        env.reset()
        conv = get_conversation_template('llama-2')
        conv.set_system_message("You are a helpful, respectful and honest assistant.")
        conv.append_message(conv.roles[0], "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]") # user
        conv.append_message(conv.roles[1], 'Ok.') # assistant
        observation = env.observation
        pre_obs = str(observation)
        conv.append_message(conv.roles[0], str(observation))
        step = 0
        # row = {'obs':observation,'action':"Null",'action_lst':"Null",'score':0}
        while step < args['env_step_limit']:
            print(observation)
            available_actions = env.get_available_actions()
            print('Available actions:', available_actions)
            # action = policy.forward(observation, available_actions)
            prompt = get_prompt(conv)
            response = llm_tgi(prompt, model, tokenizer)
            conv.append_message(conv.roles[1], response)
            action = response.replace("\n",'').split('Action:')[-1]
            # action = input("please enter the action:")
            observation, reward, done, info = env.step(action)
            conv.append_message(conv.roles[0], str(observation))
            # row = {'obs':pre_obs,'next_obs':observation,'action':action,'action_lst':available_actions,'score':reward}
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
            pre_obs = observation
            step += 1

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_step_limit", type=int, default=15)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--task_id", type=int, default=5)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="iql_webshop_1500.txt")

    parser.add_argument('--spm_path', default='IQL/spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--discount_prob', type=float, default=0.9)
    parser.add_argument('--limit_prob', type=float, default=0.5)
    parser.add_argument("--iql_path", type=str, default="IQLmodel/final_iql_webshop_twin_20.pt")
    parser.add_argument("--llm_path", type=str)

    args = parser.parse_args()
    params = vars(args)
    return params

def main():
    args = parse_args()
    # fine-tuned llama3 8B
    model_id = args["llm_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        # quantization_config=bnb_config,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        # device_map='auto',
        # max_memory=max_memory_mapping
    ).cuda()

    IQLmodel = load_model_iql(args['iql_path'], args)
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # # chat model calling API
    # client = openai.OpenAI(
    #     api_key=YOURAPIKEY,
    #     base_url="https://api.aimlapi.com/",
    # )
    # evaluate_web_chatmodel(env, client, args)

    def filter_goals(i, goal):
        if i == args['task_id']+r+2500:
        # if i == args['task_id']+r:
            return True
        return False
    for r in range(args['round']):

        env = gym.make('WebAgentTextEnv-v0', observation_mode='text', filter_goals=filter_goals, num_products=DEBUG_PROD_SIZE)
        evaluate_web_llama3finetuned_addIQL(env, model, tokenizer, IQLmodel, sbert_model, args)
        # evaluate_web_llama3finetuned(env, model, tokenizer, args)
        torch.cuda.empty_cache()
        time.sleep(3)
    
    # evaluate_web_llama3finetuned(env, model, tokenizer, args)
    # evaluate_web_llama3finetuned_addIQL(env, model, tokenizer, args)
    # env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
    # evaluate_web_human(env, args)

main()
