import argparse
from datasets import load_dataset, Dataset, load_from_disk
import jsonlines
from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import datasets

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--teacher")
    argparser.add_argument("--data")
    argparser.add_argument("--model")

    args = argparser.parse_args()
    teacher = args.teacher
    data = args.data
    model_name = args.model

    if model_name == 'qwen2.5':
        teacher_model_id = "Qwen/Qwen2.5-{}-Instruct".format(teacher)
    elif model_name == 'llama3':
        if teacher == '1B' or '3B':
            teacher_model_id = "meta-llama/Llama-3.2-{}-Instruct".format(teacher)
        else:
            teacher_model_id = "meta-llama/Llama-3.1-{}-Instruct".format(teacher)
    HF_token = 'xxxxxxx'
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)

    def write_jsonl(file_path, data):
        with jsonlines.open(file_path, 'w') as writer:
            for d in data:
                writer.write(d)
                
    def read_jsonl(file_path):
        with jsonlines.open(file_path) as reader:
            data = [d for d in reader]
        return data

    if data == 'hotpotqa':
        ds = load_dataset("hotpotqa/hotpot_qa", 'fullwiki',)['train']    
    elif data == 'triviaqa':
        ds = load_dataset('mandarjoshi/trivia_qa','rc', split='train')
    elif data == 'arc_challenge':
        ds = load_from_disk("./arc_challenge_processed")['train']
    elif data == 'gsm8k':
        ds = load_dataset('gsm8k', 'main', split='train')

    # dataset = dataset.shuffle(seed=42)
    ds = ds[:len(ds)//2] 
    # dataset = datasets.Dataset.from_list(dataset)
    ds = datasets.Dataset.from_dict(ds)

    print(ds)
    # print(dataset[0])
    llm = LLM(model=teacher_model_id, tokenizer=teacher_model_id, tensor_parallel_size=2, gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=4096)

    instructions = []
    batch_inputs = []
    ground_truths = []
    # aliases = []
    for item in ds:
        if data == 'hotpotqa':
            question = item['question']
            answer = item['answer']
        elif data == 'triviaqa':
            question = item['question']
            answer = item['answer']['value']
        elif data == 'arc_challenge': 
            question = item['question']
            messages = [{'role': 'user', 'content': '''Please solve the problem by breaking down your thinking process step by step. IMPORTANT: After explaining your reasoning, conclude with a sentence stating 'My choice is [your choice].' For example, if you choose option A, end with 'My choice is A.' \nHere is the question:''' + question.strip()}] 
            answer = item['answer']
        if data != 'arc_challenge':
            messages = [{'role': 'user', 'content': question.strip() + '\nKeep your answer in 1-3 sentences. Please provide your step-by-step thinking process: '}]
        batch_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        batch_inputs.append(batch_input)
        ground_truths.append(answer)
        instructions.append(question)


    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

    # instructions = []
    results = []

    outputs = llm.generate(batch_inputs, sampling_params)
    for output in outputs:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        # if model == 'llama':
        #     instructions.append(prompt.split('<|start_header_id|>user<|end_header_id|>\n\n')[1].split('<|eot_id|>')[0])
        # else:
        # instructions.append(prompt.split('user\n')[1].split('<|im_end|>')[0])
        results.append(generated_text)
        
    cot = []
    for instruction, response, ground_truth in zip(instructions, results, ground_truths):
        result = {"question": instruction, "answer": response, "ground_truth_response": ground_truth}
        cot.append(result)
        
    write_jsonl('./data/CoT/{}_{}_{}_cot.jsonl'.format(model_name,teacher, data), cot)

    cot = read_jsonl('./data/CoT/{}_{}_{}_cot.jsonl'.format(model_name,teacher, data))

    correct_cot = []
    cnt = 0
    for d in tqdm(cot):
        ground_truth = d['ground_truth_response'].lower()
        if data == 'arc_challenge':
            try:
                response = d['answer'].split('My choice is ')[1].split('.')[0].lower()
            except:
                continue
            for answer in ground_truth.split('. '):
                if answer.lower() in response.lower():
                    cnt += 1
                    correct_cot.append(d)
                    break
            # response = d['answer'].split('My choice is ')[1].split('.')[0].lower()
        else:
            response = d['answer'].lower()
            
        if ground_truth in response:
            cnt += 1
            correct_cot.append(d)
            
        
    print(cnt)
    print(len(cot))
    write_jsonl('./data/CoT/{}_{}_{}_cot_correct.jsonl'.format(model_name,teacher,data), correct_cot)