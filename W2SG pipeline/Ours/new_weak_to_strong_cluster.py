import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset, load_from_disk
import argparse
import re
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams
import os
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import requests
from openai import OpenAI
import transformers
import gc
import vllm
from vllm.distributed.parallel_state import destroy_model_parallel
from sentence_transformers import SentenceTransformer
import numpy as np

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data")
    argparser.add_argument("--teacher")
    argparser.add_argument("--student")
    argparser.add_argument("--method")
    argparser.add_argument("--midlevel")
    argparser.add_argument("--model")

    args = argparser.parse_args()
    data = args.data
    teacher = args.teacher
    student = args.student
    method = args.method
    midlevel = args.midlevel
    model = args.model
    
    # model = args.model


    if data == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", split='train')
    elif data == 'arc_challenge':
        ds = load_from_disk("./arc_challenge_processed")['train']
    elif data == 'hotpotqa':
        ds = load_dataset("hotpot_qa", 'fullwiki', split='train')
    elif data == 'triviaqa':
        ds = load_dataset('mandarjoshi/trivia_qa','rc', split='train')


    ds = ds[len(ds)//2:]
    ds = datasets.Dataset.from_dict(ds)
    # print(len(ds))    

    # student_model_id = "Qwen/Qwen2.5-{}-Instruct".format(student)
    
    
    if model == 'qwen2.5':
        if method == 'cascade':
            teacher_model_id = "Qwen/Qwen2.5-{}-Instruct".format(midlevel)
        else:
            teacher_model_id = "Qwen/Qwen2.5-{}-Instruct".format(teacher)
    elif model == 'llama3':
        if method == 'cascade':
            teacher_model_id = "meta-llama/Llama-3.2-{}-Instruct".format(midlevel)
        else:
            if teacher == '1B' or teacher == '3B':
                teacher_model_id = "meta-llama/Llama-3.2-{}-Instruct".format(teacher)
            else:
                teacher_model_id = "meta-llama/Llama-3.1-{}-Instruct".format(teacher)

    HF_token = 'xxxxxxx'




    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    teacher_tokenizer.padding_side = "left"
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        torch_dtype=torch.bfloat16,
        token = HF_token
    )
    
    if method == 'cascade':
        teacher_state_dict = torch.load('./model_outputs/ours_cascade_halfhalf/{}_{}_to_{}_{}/model_final.pt'.format(model,teacher, midlevel, data), map_location='cuda:0')
    else:
        teacher_state_dict = torch.load('./model_outputs/ours/{}_{}_{}/model_final.pt'.format(model,teacher, data), map_location='cuda:0')

    teacher_new_state_dict = {}
    for key in teacher_state_dict.keys():
        if 'module.' not in key:
            new_key = key
        else:
            new_key = key.split('module.')[1]
        teacher_new_state_dict[new_key] = teacher_state_dict[key]
    

    teacher_model.load_state_dict(teacher_new_state_dict)
    if method == 'cascade':
        teacher_model_path = './model_outputs/for_vllm/ours_cascade_halfhalf/{}_{}_to_{}_{}_model/'.format(model,teacher, midlevel, data)
        teacher_tokenizer_path = './model_outputs/for_vllm/ours_cascade_halfhalf/{}_{}_to_{}_{}_tokenizer/'.format(model,teacher, midlevel, data)
    else:
        teacher_model_path = './model_outputs/for_vllm/ours/{}_{}_{}_model/'.format(model,teacher, data)
        teacher_tokenizer_path = './model_outputs/for_vllm/ours/{}_{}_{}_tokenizer/'.format(model,teacher, data)


    if not os.path.exists(teacher_model_path):
        teacher_model.save_pretrained(teacher_model_path)
        teacher_tokenizer.save_pretrained(teacher_tokenizer_path)
    
    
    tokenizer_path = teacher_tokenizer_path
    tokenizer = teacher_tokenizer
    model_path = teacher_model_path
    llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.6, tensor_parallel_size = 2)  
    
    def batch_inference(inputs):

        batch_inputs = []
        for item in inputs:
            messages = [{'role': 'user', 'content': item}]
            batch_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            batch_inputs.append(batch_input)

        sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=4096)
        
        instructions = []
        results = []
        
        outputs = llm.generate(batch_inputs, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            if model == 'qwen2.5':
                instructions.append(prompt.split('user\n')[1].split('<|im_end|>')[0])
            elif model == 'llama3':
                instructions.append(prompt.split('<|start_header_id|>user<|end_header_id|>\n\n')[1].split('<|eot_id|>')[0])
            results.append(generated_text)
        
        return instructions, results



    if data == 'gsm8k':
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        embedding_model = INSTRUCTOR('hkunlp/instructor-large')


    def calculate_similarity(question, responses, target, threshold=0.9):
        instruction_embedding = 'Represent the paragraph for the question "{}":'
        if data == 'gsm8k':
            responses = [item for item in responses]
            target = [target]
        else:   
            responses = [[instruction_embedding.format(question), item] for item in responses]
            target = [[instruction_embedding.format(question), target]]
        
        with torch.no_grad():
            responses_vector = embedding_model.encode(responses)
            target_vector = embedding_model.encode(target)
            similarity = cosine_similarity(responses_vector, target_vector)
        # print(similarity)
        return (similarity > threshold)



    def clustering(question, responses):
        clustered = []
        temp_outputs = responses.copy()
        
        sizes = []
        while len(temp_outputs) > 0:
            output = temp_outputs[0]
            similar_idx = calculate_similarity(question, temp_outputs, output, 0.9)
            size = int(similar_idx.sum())
            if size == 1:
                temp_outputs = temp_outputs[1:]
                continue
            sizes.append(size)
            clustered.append({
                # NUM_TRY = 100
                'response':  output,
                'size': size,
            })
            fetch_idx = (similar_idx == False).nonzero()[0]
            if len(fetch_idx) == 0:
                break
            temp_outputs = [temp_outputs[i] for i in fetch_idx]
        return clustered, sizes


    def clustering_on_answer(responses):
        clustered = []
        temp_outputs = responses.copy()
        
        sizes = []
        while len(temp_outputs) > 0:
            output = temp_outputs[0]
            try:
                if data == 'gsm8k':
                    extracted_output = temp_outputs[0].split('#### ')[1].strip()
                elif data == 'arc_challenge':
                    extracted_output = temp_outputs[0].split('My choice is ')[1].strip()
            except:
                temp_outputs = temp_outputs[1:]
                continue
            a = []
            for item in temp_outputs:
                try:
                    if data == 'gsm8k':
                        a.append(item.split('#### ')[1].strip())
                    elif data == 'arc_challenge':
                        a.append(item.split('My choice is ')[1].strip())
                except:
                    a.append('N/A')
            a = np.array(a)
            similar_idx = (a == extracted_output)
            
            size = int(similar_idx.sum())
            if size == 1:
                temp_outputs = temp_outputs[1:]
                continue
            sizes.append(size)
            clustered.append({
                # NUM_TRY = 100
                'response':  output,
                'size': size,
            })
            fetch_idx = (similar_idx == False).nonzero()[0]
            if len(fetch_idx) == 0:
                break
            temp_outputs = [temp_outputs[i] for i in fetch_idx]
        return clustered, sizes
        



    key = 'question'

    if method == 'cascade':
        if data == 'hotpotqa' or data == 'triviaqa':
            parts = int(0.2*len(ds))
            raw_inputs = ds[key][parts//2:parts] ##last half of data for 3B -7B
        else:
            raw_inputs = ds[key][len(ds)//2:]
    else:
        if data == 'hotpotqa' or data == 'triviaqa':
            parts = int(0.2*len(ds))
            raw_inputs = ds[key][:parts]
        else:
            raw_inputs = ds[key]
    inputs = [item for item in raw_inputs for _ in range(100)]
    
    print(len(inputs))
    print('Start batch inference...')
    
    for r in tqdm(range(0, len(inputs), 150000)):
        chunk = inputs[r:r + 150000]
        raw_instructions, raw_results = batch_inference(chunk)

        instructions = []
        sampled_responses = []

        for i in range(0, len(raw_instructions), 100):
            instruction = raw_instructions[i]
            instructions.append(instruction)
            responses = raw_results[i:i + 100]
            sampled_responses.append(responses)

        
        clusters = []
        final_responses = []
        
        
        
        
        for instruction, sampled_results in zip(instructions, sampled_responses):
            ##add clustering and uncertainty extraction here
            if data == 'gsm8k' or data == 'arc_challenge':
                clustered, sizes = clustering_on_answer(sampled_results)
            else:
                clustered, sizes = clustering(instruction, sampled_results)
            # print(clustered)
            clusters.append(clustered)
            sorted_clustered = sorted(clustered, key=lambda x: x['size'], reverse=True)
            try:
                final_response = sorted_clustered[0]['response']
                final_responses.append(final_response)
            except:
                print(clustered)
                final_responses.append('N/A')
                continue

            
        for question, sampled_response, cluster, final_response in zip(instructions, sampled_responses, clusters, final_responses):
            if final_response != 'N/A':
                result = {"question": question, "sampled_responses": sampled_response, "clusters": cluster, 'final_response': final_response}
                # print(result)
                if method == 'cascade':
                    with open('./data/ours_cascade_halfhalf/{}/{}_{}_{}_sampled_clustered.jsonl'.format(data, model,midlevel, data), 'a') as f:
                        f.write(json.dumps(result) + '\n')
                else: 
                    with open('./data/ours/{}/{}_{}_{}_sampled_clustered.jsonl'.format(data, model,teacher,data), 'a') as f:
                        f.write(json.dumps(result) + '\n')

            
    print('Clusters generated and saved.')