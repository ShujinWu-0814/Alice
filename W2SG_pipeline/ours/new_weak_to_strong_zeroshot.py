import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset
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
import jsonlines

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data")
    argparser.add_argument("--teacher")
    argparser.add_argument("--student")
    argparser.add_argument("--method")
    argparser.add_argument("--midlevel")
    argparser.add_argument('--model')

    args = argparser.parse_args()
    data = args.data
    teacher = args.teacher
    student = args.student
    method = args.method
    midlevel = args.midlevel
    model = args.model


        
    if model =='qwen2.5':
        student_model_id = "Qwen/Qwen2.5-{}-Instruct".format(student)
    elif model == 'llama3':
        if student == '1B' or student == '3B':
            student_model_id = "meta-llama/Llama-3.2-{}-Instruct".format(student)
        else:
            student_model_id = "meta-llama/Llama-3.1-{}-Instruct".format(student)
            
    # teacher_model_id = "Qwen/Qwen2.5-{}-Instruct".format(teacher)
    HF_token = 'xxxxxxx'



    student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
    student_tokenizer.padding_side = "left"
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        torch_dtype=torch.bfloat16,
        token = HF_token
    )


    def batch_inference(modell, inputs):
        if modell == 'llama':
            tokenizer_path = './Meta-Llama-3-70B-Instruct'
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=HF_token)
            model_path = './Meta-Llama-3-70B-Instruct'
        elif modell == 'student':
            tokenizer_path = student_model_id
            tokenizer = student_tokenizer
            model_path = student_model_id
        llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.70, tensor_parallel_size = 4)  
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


    if method == 'cascade':
        with jsonlines.open('./data/ours_cascade_halfhalf/{}/{}_{}_{}_uncertainty.jsonl'.format(data,model,midlevel,data)) as reader:
            ds = [d for d in reader]
    else:
        with jsonlines.open('./data/ours/{}/{}_{}_{}_uncertainty.jsonl'.format(data,model,teacher,data)) as reader:
            ds = [d for d in reader]
   
   
    questions = [ds[i]['question'] for i in range(len(ds))]
    #zero shot
    q, a = batch_inference('student', questions)
    
    for question, answer in zip(q,a):
        if method == 'cascade':
            with open('./data/ours_cascade_halfhalf/{}/{}_{}_{}_zeroshot.jsonl'.format(data,model,student, data), 'a') as f:
                f.write(json.dumps({'question': question, 'zero_shot_student_response': answer}) + '\n')
        else:
            with open('./data/ours/{}/{}_{}_{}_zeroshot.jsonl'.format(data,model,student, data), 'a') as f:
                f.write(json.dumps({'question': question, 'zero_shot_student_response': answer}) + '\n')
    

    print('Zero shot inference for student model done and saved.')

        
