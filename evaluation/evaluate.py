import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import argparse
import re
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data")
    argparser.add_argument("--teacher")
    argparser.add_argument("--student")
    argparser.add_argument("--method")
    argparser.add_argument("--original-wts-type")
    argparser.add_argument("--ours-wts-type")
    argparser.add_argument("--model")

    args = argparser.parse_args()
    data = args.data
    teacher = args.teacher
    student = args.student
    method = args.method
    original_wts_type = args.original_wts_type
    ours_wts_type = args.ours_wts_type
    model_name = args.model
    
    if model_name == 'qwen2.5':
        model_id = "Qwen/Qwen2.5-{}-Instruct".format(student)
    elif model_name == 'llama3':
        if student == '8B':
            model_id = "meta-llama/Llama-3.1-{}-Instruct".format(student)
        else:
            model_id = "meta-llama/Llama-3.2-{}-Instruct".format(student)
            
    HF_token = 'xxxxxxxx'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token = HF_token
    ).to('cuda')


    if method == 'naive': ##zero-shot without any training
        model_path = model_id
        tokenizer_path = model_id
    if method == 'base':
        if data == 'gsm8k':
            state_dict_path = './model_outputs/base/{}_{}_{}/model_final.pt'.format(model_name,student, data)
            model_path = './model_outputs/for_vllm/base/{}_{}_{}_model/'.format(model_name,student, data)
            tokenizer_path = './model_outputs/for_vllm/base/{}_{}_{}_tokenizer/'.format(model_name,student, data)
    elif method == 'original-wts':
        if original_wts_type == 'cascade':
            state_dict_path = './model_outputs/original_cascade/{}_{}_to_{}_{}/model_final.pt'.format(model_name,teacher, student, data)
            model_path = './model_outputs/for_vllm/original_cascade/{}_{}_to_{}_{}_model/'.format(model_name,teacher, student, data)
            tokenizer_path = './model_outputs/for_vllm/original_cascade/{}_{}_to_{}_{}_tokenizer/'.format(model_name,teacher, student, data)
        else:
            state_dict_path = './model_outputs/original/{}_{}_to_{}_{}_{}teacher/model_final.pt'.format(model_name,teacher, student, data, original_wts_type)
            model_path = './model_outputs/for_vllm/original/{}_{}_to_{}_{}_{}teacher_model/'.format(model_name,teacher, student, data, original_wts_type)
            tokenizer_path = './model_outputs/for_vllm/original/{}_{}_to_{}_{}_{}teacher_tokenizer/'.format(model_name,teacher, student, data, original_wts_type)
    elif method == 'ours-wts':
        if ours_wts_type == 'singleturn':
            state_dict_path = './model_outputs/ours/{}_{}_to_{}_{}/model_final.pt'.format(model_name,teacher, student, data)
            model_path = './model_outputs/for_vllm/ours/{}_{}_to_{}_{}_model/'.format(model_name,teacher, student, data)
            tokenizer_path = './model_outputs/for_vllm/ours/{}_{}_to_{}_{}_tokenizer/'.format(model_name,teacher, student, data)
        elif ours_wts_type == 'cascade':
            state_dict_path = './model_outputs/ours_{}_halfhalf/{}_{}_to_{}_{}/model_final.pt'.format(ours_wts_type, model_name,teacher, student, data)
            model_path = './model_outputs/for_vllm/ours_{}_halfhalf/{}_{}_to_{}_{}_model/'.format(ours_wts_type,model_name,teacher, student, data)
            tokenizer_path = './model_outputs/for_vllm/ours_{}_halfhalf/{}_{}_to_{}_{}_tokenizer/'.format(ours_wts_type,model_name,teacher, student, data)
        else:
            state_dict_path = './model_outputs/ours_{}/{}_{}_to_{}_{}/model_final.pt'.format(ours_wts_type,model_name,teacher, student, data)
            model_path = './model_outputs/for_vllm/ours_{}/{}_{}_to_{}_{}_model/'.format(ours_wts_type,model_name,teacher, student, data)
            tokenizer_path = './model_outputs/for_vllm/ours_{}/{}_{}_to_{}_{}_tokenizer/'.format(ours_wts_type,model_name,teacher, student, data)
    
    if method != 'naive':
        state_dict = torch.load(state_dict_path, map_location='cuda')
        new_state_dict = {}
        for key in state_dict.keys():
            if 'module.' not in key:
                new_key = key
            else:
                new_key = key.split('module.')[1]
            new_state_dict[new_key] = state_dict[key]
            
        model.load_state_dict(new_state_dict)

 
    if not os.path.exists(model_path):
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)


 
    if data == 'gsm8k':
        ds = datasets.load_dataset("openai/gsm8k", "main", split='test')
    elif data == 'arc_challenge':
        ds = datasets.load_from_disk("./arc_challenge_processed")['test']
    elif data == 'hotpotqa':
        ds = datasets.load_dataset("hotpot_qa", 'fullwiki', split='validation')
    elif data == 'triviaqa':
        ds = datasets.load_dataset('trivia_qa', 'rc', split='validation')

        


    def get_response(inputs):
        llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.6, tensor_parallel_size = 2)
        texts = []
        for message in inputs:
            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=4096)
        
        results = []
        outputs = llm.generate(texts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results


    def extract_boxed_content(input_string):
        start_marker = r'\boxed{'
        start_idx = input_string.find(start_marker)
        if start_idx == -1:
            return None  # No \boxed{ found

        idx = start_idx + len(start_marker)
        content_start = idx
        brace_count = 1  # Starting with the '{' after '\boxed'

        while idx < len(input_string):
            char = input_string[idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace for \boxed{
                    content_end = idx  # We don't include this closing brace
                    break
            idx += 1
        else:
            # If we exit the loop without breaking, braces are unbalanced
            return None

        # Extract the content between content_start and content_end
        content = input_string[content_start:content_end]
        return content




    result = 0
    cnt = 0
    inputs = []
    correct_answers = []
    questions = []
    for item in tqdm(ds):
        if data == 'gsm8k':
            messages = [{'role': 'user', 'content': item['question']}]
            correct_answer = item['answer'].split('####')[1].strip()
        elif data == 'triviaqa':
            messages = [{'role': 'user', 'content': item['question']}]
            correct_answer = item['answer']['normalized_aliases']
        elif data == 'hotpotqa':
            messages = [{'role': 'user', 'content': item['question']}]
            correct_answer = item['answer']
        else:
            messages = [{'role': 'user', 'content': item['question']}]
            correct_answer = item['answer'].split('. ')
        inputs.append(messages)
        correct_answers.append(correct_answer)
        questions.append(item['question'])

    responses = get_response(inputs)
    # print(len(responses), len(correct_answers))
    for question, response, correct_answer in zip(questions, responses, correct_answers):
        if data == 'gsm8k':
            with open('./{}_evaluation_results_{}.jsonl'.format(student, method), 'a') as f:
                f.write(json.dumps({'question': question, 'response': response, 'correct_answer': correct_answer}) + '\n')
            try:
                response = response.split('####')[1].strip()
                cnt += 1
            except:
                print('No match found')
                continue
            if response == correct_answer:
                result += 1
        elif data == 'hotpotqa':
            cnt += 1
            with open('./{}_evaluation_results_{}.jsonl'.format(student, method), 'a') as f:
                f.write(json.dumps({'question': question, 'response': response, 'correct_answer': correct_answer}) + '\n')
            if correct_answer.lower() in response.lower():
                result += 1
        elif data == 'triviaqa':
            cnt += 1
            with open('./{}_evaluation_results_{}.jsonl'.format(student, method), 'a') as f:
                f.write(json.dumps({'question': question, 'response': response, 'correct_answer': correct_answer}) + '\n')
            for answer in correct_answer:
                if answer.lower() in response.lower():
                    result += 1
                    break
        elif data == 'arc_challenge':
            with open('./{}_evaluation_results_{}.jsonl'.format(student, method), 'a') as f:
                f.write(json.dumps({'question': question, 'response': response, 'correct_answer': correct_answer}) + '\n')
            if original_wts_type == 'cot' or method == 'ours-wts':
                try:
                    response = response.split('My choice is')[1].strip().split('.')
                    response = list(filter(lambda x: x.strip(), response))
                    cnt += 1
                except:
                    print('No match found')
                    continue
            else:
                cnt += 1
                response = response.split('.')
            
            for i in range(min(len(response), len(correct_answer))):
                if correct_answer[i].lower().strip() in response[i].lower().strip():
                    result += 1
                    break
            

        
    print(result)
    print(cnt)
    print(len(ds))
    correct_rate = (result/cnt)*100
    print(correct_rate)
