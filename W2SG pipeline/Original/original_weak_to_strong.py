import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset, load_from_disk
import argparse
import re
from tqdm import tqdm
import json
import os
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data")
    argparser.add_argument("--teacher")
    argparser.add_argument("--type")
    argparser.add_argument("--midlevel")
    argparser.add_argument("--model")

    args = argparser.parse_args()
    data = args.data
    teacher = args.teacher
    typee = args.type
    midlevel = args.midlevel
    model_name = args.model
 
    if data == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", split='train')
    elif data == 'arc_challenge':
        ds = load_from_disk('./arc_challenge_processed')['train']
    elif data == 'hotpotqa':
        ds = load_dataset("hotpot_qa", 'fullwiki', split='train')
    elif data == 'triviaqa':
        ds = load_dataset('mandarjoshi/trivia_qa','rc', split='train')

    ds = ds[len(ds)//2:]
    ds = datasets.Dataset.from_dict(ds)

    
    

    if typee == 'cascade_1' or typee == 'cascade_2':
        if model_name =='qwen2.5':
            model_id = "Qwen/Qwen2.5-{}-Instruct".format(midlevel)
        elif model_name == 'llama3':
            if midlevel == '8B':
                model_id = "meta-llama/Llama-3.1-{}-Instruct".format(midlevel)
            else:
                model_id = "meta-llama/Llama-3.2-{}-Instruct".format(midlevel)
        # ds = ds[len(ds)//2:]
        # # ds = datasets.Dataset.from_dict(ds)
    else:
        if model_name == 'qwen2.5':
            model_id = "Qwen/Qwen2.5-{}-Instruct".format(teacher)
        elif model_name == 'llama3':
            if teacher == '8B':
                model_id = "meta-llama/Llama-3.1-{}-Instruct".format(teacher)
            else:
                model_id = "meta-llama/Llama-3.2-{}-Instruct".format(teacher)
    HF_token = 'xxxxxxxx'



    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token = HF_token
    ).to('cuda')
    tokenizer.add_special_tokens({"pad_token": "<unk>"})
    
    
    if typee == 'qa':
        state_dict = torch.load('./model_outputs/original/{}_{}_{}/model_final.pt'.format(model_name,teacher, data),map_location='cuda:0')
    elif typee == 'cot' or typee == 'cascade_1':
        state_dict = torch.load('./model_outputs/ours/{}_{}_{}/model_final.pt'.format(model_name,teacher, data),map_location='cuda:0')
    elif typee == 'cascade_2':
        state_dict = torch.load('./model_outputs/original_cascade/{}_{}_to_{}_{}/model_final.pt'.format(model_name,teacher,midlevel, data),map_location='cuda:0')
    new_state_dict = {}
    
    for key in state_dict.keys():
        new_key = key.split('module.')[1]
        new_state_dict[new_key] = state_dict[key]
        
    model.load_state_dict(new_state_dict)

    if typee == 'qa':
        model_path = './model_outputs/for_vllm/original/{}_{}_{}_model/'.format(model_name,teacher, data)
        tokenizer_path = './model_outputs/for_vllm/original/{}_{}_{}_tokenizer/'.format(model_name,teacher, data)
    elif typee == 'cot' or typee == 'cascade_1':
        model_path = './model_outputs/for_vllm/ours/{}_{}_{}_model/'.format(model_name,teacher, data)
        tokenizer_path = './model_outputs/for_vllm/ours/{}_{}_{}_tokenizer/'.format(model_name,teacher, data)
    elif typee == 'cascade':
        model_path = './model_outputs/for_vllm/original_cascade/{}_{}_to_{}_{}_model/'.format(model_name,teacher,midlevel, data)
        tokenizer_path = './model_outputs/for_vllm/original_cascade/{}_{}_to_{}_{}_tokenizer/'.format(model_name,teacher,midlevel, data)
        
    if not os.path.exists(model_path):
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
        




    def batch_inference(inputs):
        llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.6, tensor_parallel_size = 2)  
        batch_inputs = []
        for item in inputs:

            messages = [{'role': 'user', 'content': item}]
            batch_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            batch_inputs.append(batch_input)

        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
        
        instructions = []
        results = []
        outputs = llm.generate(batch_inputs, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            if model_name == 'qwen2.5':
                instructions.append(prompt.split('user\n')[1].split('<|im_end|>')[0])
            elif model_name == 'llama3':
                instructions.append(prompt.split('<|start_header_id|>user<|end_header_id|>\n\n')[1].split('<|eot_id|>')[0])
            results.append(generated_text)

        return instructions, results





    key = 'question'

    if typee == 'cascade_2':
        if data == 'hotpotqa' or data == 'triviaqa':
            parts = int(0.2*len(ds))
            raw_inputs = ds[key][parts//2:parts]
        else:
            raw_inputs = ds[key][len(ds)//2:]
    if typee == 'cascade_1':
        if data == 'hotpotqa' or data == 'triviaqa':
            parts = int(0.2*len(ds))
            raw_inputs = ds[key][:parts//2]
        else:
            raw_inputs = ds[key][:len(ds)//2]
    else:
        if data == 'hotpotqa' or data == 'triviaqa':
            parts = int(0.2*len(ds))
            raw_inputs = ds[key][:parts]
        else:
            raw_inputs = ds[key]
    # raw_inputs = ds[key]

        
        
    print('Starting inference...')
    
    instructions, results = batch_inference(raw_inputs)

    print('Finished inference, writing to file...')


    # if typee == 'qa':
    if typee == 'cascade_1':
        path = './data/original_cascade/{}_{}_{}.jsonl'.format(model_name,teacher,data)
    elif typee == 'cascade_2':
        path = './data/original_cascade/{}_{}_{}.jsonl'.format(model_name,midlevel,data)        
    else:
        path = './data/original/{}_{}_{}_{}teacher.jsonl'.format(model_name,teacher,data,typee)

    with open(path, 'a') as f:
        for instruction, response in zip(instructions, results):
            result = {"question": instruction, "answer": response}
            f.write(json.dumps(result) + '\n')
        
        
        