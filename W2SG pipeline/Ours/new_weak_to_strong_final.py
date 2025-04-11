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
    HF_token = 'xxxxx'



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
        llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.60, tensor_parallel_size = 4)  
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




    prompt_to_student = '''
    You will be provided with a question,  a response from yourself, and a response from another model and the uncertainty statement from that model as well.
    You should take another model's response and the uncertainty statement as references to help you provide a final response to the question. You are also required to provide your step-by-step solution to the question.
    Here is the question: {}
    Here is your original answer: {}
    Here is another model's answer: {}
    Here is another model's uncertainty statement: {}

    Note: You should not mention the model's response or the uncertainty statement in your new response. Just simply output your final response to the given question. Do not explain about why you choose to keep your original response or give a new one.
    IMPORTANT: You must provide an answer to the question. You should not say 'I don't know' or 'I am not sure'. You should always try your best to provide an answer at your best knowledge.'''


    prompt_to_student_gsm8k = '''
    You will be provided with a question,  a response from yourself, and a response from another model and the uncertainty statement from that model as well.
    You should refer to another model's response and the uncertainty statement to update your original response, if necessary, based on your own knowledge.
    Here is the question: {}
    Here is your original answer: {}
    Here is another model's answer: {}
    Here is another model's uncertainty statement: {}

    Note: You should not mention your original response, the model's response or the uncertainty statement in your new response. If you think your original response is correct, just restate it and format it as requested later. If you think it should be revised, provide the revised response and format it as requested as well. DO NOT explain why you choose to keep the original response or give a revised one.
    Format Requirement: You should add a small part at the very end of your response to report the final answer in number only in this format: '#### <answer>'.
    Here is an example of your response: 'Each sandwich needs 3 slices of bread for a total of 3*5=<<3*5=15>>15 slices of bread\n#### 15'.'''


    prompt_to_student_arc = '''
    You will be provided with a question,  a response from yourself, and a response from another model and the uncertainty statement from that model as well.
    You should refer to another model's response and the uncertainty statement to update your original response, if necessary, based on your own knowledge.
    Here is the question: {}
    Here is your original answer: {}
    Here is another model's answer: {}
    Here is another model's uncertainty statement: {}

    Note: You should not mention your original response, the model's response or the uncertainty statement in your new response. If you think your original response is correct, just restate it and format it as requested later. If you think it should be revised, provide the revised response and format it as requested as well. DO NOT explain why you choose to keep the original response or give a revised one.
    Format Requirement: You should add a small part at the very end of your response to report your final choice in this format: 'My choice is [your choice].' For example, if your final choice is option A, end with 'My choice is A.'
    Here is an example of your response: 'The Moon's constant rotation (lunar phases) completes in the same amount of time it takes to orbit the Earth.\nMy choice is A.'.'''


    if method == 'cascade':
        with jsonlines.open('./data/ours_cascade_halfhalf/{}/{}_{}_{}_uncertainty.jsonl'.format(data,model,midlevel,data)) as reader:
            ds = [d for d in reader]
        
        with jsonlines.open('./data/ours_cascade_halfhalf/{}/{}_{}_{}_zeroshot.jsonl'.format(data,model,student, data)) as reader:
            zeroshot = [d for d in reader]
    else:
        with jsonlines.open('./data/ours/{}/{}_{}_{}_uncertainty.jsonl'.format(data,model,teacher,data)) as reader:
            ds = [d for d in reader]
    
        with jsonlines.open('./data/ours/{}/{}_{}_{}_zeroshot.jsonl'.format(data,model,student, data)) as reader:
            zeroshot = [d for d in reader]
        

    instruction_to_student = []
    to_save = []
    for i, item in enumerate(ds):
        if data == 'gsm8k':
            instruction_to_student.append(prompt_to_student_gsm8k.format(item['question'], zeroshot[i]['zero_shot_student_response'], item['final_response'], item['uncertainty']))
        elif data == 'arc_challenge':
            instruction_to_student.append(prompt_to_student_arc.format(item['question'], zeroshot[i]['zero_shot_student_response'], item['final_response'], item['uncertainty']))
        else:
            instruction_to_student.append(prompt_to_student.format(item['question'], zeroshot[i]['zero_shot_student_response'], item['final_response'], item['uncertainty']))
        to_save.append({'question': item['question'], 'label':'new'})
    
    instruction_to_summarize_uncertainty, new_responses = batch_inference('student', instruction_to_student)
    
    for i, new_response in enumerate(new_responses):
        to_save[i]['answer'] = new_response
    
    for item in to_save:
        if method == 'cascade':
            with open('./data/ours_cascade_halfhalf/{}/{}_{}_to_{}_{}.jsonl'.format(data,model,midlevel, student, data), 'a') as f:
                f.write(json.dumps(item) + '\n')
        else:
            with open('./data/ours/{}/{}_{}_to_{}_{}.jsonl'.format(data,model,teacher, student, data), 'a') as f:
                f.write(json.dumps(item) + '\n')
       

    print('Full pipeline completed and new data saved.')

        
