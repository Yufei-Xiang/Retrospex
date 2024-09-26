import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import gc


def get_llama3_7b_response(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    if input_ids.shape[1] >= 4048:
        messages = messages[-10:]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    print(result)
    return result

def get_llama3_7b_response_multi(model, tokenizer, messages, k):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print(input_ids.shape)
    if input_ids.shape[1] >= 4048:
        messages = messages[-10:]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=k
        )
    # print(outputs.keys())
    # print(outputs['scores'])
    responses = []
    for i in range(k):
        response = outputs[i][input_ids.shape[-1]:]
        result = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(result)
    # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(result)
    gc.collect()
    torch.cuda.empty_cache()
    return responses

def cal_prob(target_text,messages,model,tokenizer):
    temp_messages = messages[:]
    temp_messages.append({'role':'assistant','content':target_text})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    encodings = tokenizer.apply_chat_template(temp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # print(input_ids.shape)
    # print(input_ids)
    # print(encodings.shape)
    # print(encodings)

    labels = encodings.clone()
    labels[:,:input_ids.shape[1]] = -100
    # print(labels)
    # print(labels)

    #计算生成text的概率
    with torch.no_grad():
        outputs = model.forward(encodings, labels=labels)
        loss = outputs[0]
        text_prob=torch.exp(-loss)#**(len(target_text))
    return text_prob

def get_actions_origin(predictions):
    action_lst = []
    for pred in predictions:
        pred_action = pred.replace("\n",'').split('Action:')[-1]
        action_lst.append(pred_action.lower())
    return action_lst

def MappingValidActionList(predictions, validActions, sbert_model, k):
    action_lst = []
    remain = []
    if validActions == []:
        for pred in predictions[:k]:
            pred_action = pred.replace("\n",'').split('Action:')[-1]
            action_lst.append(pred_action)
        return action_lst

    if len(validActions) < k:
        return validActions

    for pred in predictions[:k]:
        pred_action = pred.replace("\n",'').split('Action:')[-1]
        if pred_action.lower() in validActions:
            if pred_action.lower() not in action_lst:
                action_lst.append(pred_action.lower())
        else:
            if pred_action.lower() not in remain:
                remain.append(pred_action.lower())
    lr = len(remain)
    print(remain)
    if lr != 0 and sbert_model:
        # text = remain+validActions
        # embeddings = sbert_model.encode(text, show_progress_bar=False)
        # for i in range(lr):
        #     similarity = cosine_similarity([embeddings[i]], embeddings[lr:])
        #     s = similarity.tolist()[0]
        #     sorted_id = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        #     for j in sorted_id:
        #         if validActions[j] not in action_lst:
        #             action_lst.append(validActions[j])
        #             break

        pred_vectors = sbert_model.encode(predictions[:k], batch_size=k, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)
        sum_similarities = similarity_matrix.sum(axis=0)
        top_indices = np.argpartition(sum_similarities, -k)[-k:]
        count = 0
        for i in range(k):
            action = validActions[top_indices[i]]
            if action not in action_lst:
                action_lst.append(action)
                count += 1
            if count == lr:
                break
    # if lr == 0 and len(action_lst)<k:
    #     choose_set = set(validActions)-set(action_lst)
    #     action_lst += random.choices(list[choose_set], k=k-len(action_lst))
    gc.collect()
    torch.cuda.empty_cache()
    return action_lst

# model_id = "/webdav/MyData/agentlm3"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # quantization_config=bnb_config,
#     attn_implementation="eager",
#     torch_dtype=torch.float16,
#     device_map='auto'
# )
# # # model = None
# messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful, respectful and honest assistant."
#         },
#         {
#             "role": "user",
#             "content": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]"
#         },
#         {
#             "role": "assistant",
#             "content": "Ok."
#         },
#         {
#             "role": "user",
#             "content": "WebShop [SEP] Instruction: [SEP] i am looking for an easy to install white antler chandelier with 18 antlers and 9 lights, and price lower than 410.00 dollars [SEP] Search"
#         },
#         {
#             "role": "assistant",
#             "content": "Thought:\nI think I should use the search bar to look for the product I need.\n\nAction:\nsearch[easy to install white antler chandelier 18 antlers 9 lights price < 410.00]"
#         },
#         {
#             "role": "user",
#             "content": "Instruction: [SEP] i am looking for an easy to install white antler chandelier with 18 antlers and 9 lights, and price lower than 410.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B093352T72 [SEP] Durahonn White Antler Chandelier, Retro Resin Deer Horn Pendant Light, E12 Lamp Holder, Commercial and Home Lighting for Cafes, Bars, Restaurants, Living Rooms (15 Antlers + 9 Lights) [SEP] $379.99 [SEP] B092ZPKVFR [SEP] Durahonn Antler Chandelier, 9 Lights E12 Bulbs, Brown Resin Deer Horn Chandelier, Retro Antler Pendant Light for Kitchen, Bar, Living Room, Dining Room (15 Antlers + 9 Lights) [SEP] $379.99 to $219.84 [SEP] B07VWTDCJT [SEP] Antler Chandelier Lighting,6 Light Antler Deer Chandelier for Living Room Dining Room Balcony Bedroom Cafe Bar Cabin Hanging Light Fixtures [SEP] $279.99 to $169.62 [SEP] B092MKQVW2 [SEP] Durahonn Vintage Antler Chandelier, E12 6 Lights ,Resin Deer Horn Pendant Light for Kitchen, Cafe, Living Room, Restaurant, Bar, Rustic Antler Lamp (9 Antlers + 6 Lights) [SEP] $209.99 [SEP] B09B3GMTW9 [SEP] Large Antler Chandelier 12 Lights, BIGMAII Cabin Retro Faux Antler Light Fixture Rustic Resin Pendant Light Farmhouse Candle Style for Living Room, Brown [SEP] $100.0 [SEP] B09G5RHTXD [SEP] Hubrin Rustic Antler Chandelier, Resin Deer Horn Pendant Light , Antler Light Fixtures 9 Light Brown E12 Candle Style for Home Store (9 lamp arms + 6arms) [SEP] $378.99 [SEP] B0989TJF5G [SEP] Durahonn Resin Antler Chandelier Pendant Ceiling Lighting Antler Decor Lamp Retro Country Style Home Decoration Chandelier for Coffee Shop, Living Room, Clothing Store (9 Lamp Antlers + 9 Antlers) [SEP] $429.99 [SEP] B091H7Q9WB [SEP] S Y, Wall Light Branch 2 Lights Antlers Sconce Rustic Style Metal and Resin Lamp Indoor Decoration [SEP] $98.99 [SEP] B076HKKR6L [SEP] Antler Chandelier 6 Light Vintage Style Large Antler Light Fixture Ceiling 24.5 Inches Diameter X 15.75 Inches High for Living Room Dining Room Big Deer Hanging Lighting. [SEP] $167.0 [SEP] B0967VS85P [SEP] 20\"Antler Wall Sconce,Rustic Wall Light Branch Resin Deer Lamp Fixture Wall Lamp Vintage Deer Horn Decor Lighting,Antler Chandelier Creative Retro Light Up Hallway Wall Mount Light Indoor Decoration [SEP] $79.99"
#         },
#         {
#             "role": "assistant",
#             "content": "Thought:\nI think I should click on the product that matches my requirements, which is B093352T72.\n\nAction:\nclick[B093352T72]"
#         },
#         {
#             "role": "user",
#             "content": "Instruction: [SEP] i am looking for an easy to install white antler chandelier with 18 antlers and 9 lights, and price lower than 410.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 12 antlers + 9 lights [SEP] 15 antlers + 9 lights [SEP] 18 antlers + 9 lights [SEP] Durahonn White Antler Chandelier, Retro Resin Deer Horn Pendant Light, E12 Lamp Holder, Commercial and Home Lighting for Cafes, Bars, Restaurants, Living Rooms (15 Antlers + 9 Lights) [SEP] Price: $379.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
#         }
# ]

# target_text = "click[18 antlers + 9 lights]"
# target_text = "click[return to search]"
# p = cal_prob(target_text, messages, model, tokenizer)
# print(p)
# for i in range(100):
#     responses = get_llama3_7b_response_multi(model, tokenizer, messages, 4)
#     print(responses)
# a = ['click[a]']
# b = ['click[a]','click[b]','click[c]','click[d]']
# choose_set = set(b)-set(a)
# print(choose_set)
# a += random.choices(list(choose_set), k=2)
# print(a)

# from vllm import LLM, SamplingParams

# llm = LLM(model="/home/yfwang/xyf/agentlm3")  # Name or path of your model
# terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

# sampling_params = SamplingParams(n=2,temperature=0.8, top_p=0.95, use_beam_search=True,stop_token_ids=terminators)
# outputs = llm.generate(messages, sampling_params)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
