from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.data.distributed import DistributedSampler
import random
import datasets
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import os
import random
import argparse


accelerator = Accelerator(gradient_accumulation_steps=16)
argparser = argparse.ArgumentParser()

argparser.add_argument("--data")
argparser.add_argument("--method")
argparser.add_argument("--original-wts-type")
argparser.add_argument("--ours-wts-type")
argparser.add_argument("--teacher")
argparser.add_argument("--student")
argparser.add_argument("--model")


args = argparser.parse_args()
data = args.data
method = args.method
teacher = args.teacher
student = args.student
original_wts_type = args.original_wts_type
ours_wts_type = args.ours_wts_type
model = args.model



if method =='base': 
    if data == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", split='train')
        ds = ds[len(ds)//2:]
        ds = datasets.Dataset.from_dict(ds)
    elif data == 'arc_challenge':
        ds = load_from_disk("./arc_challenge_processed")['train']
        ds = ds[len(ds)//2:]
        ds = datasets.Dataset.from_dict(ds)
    elif data == 'hotpotqa':
        ds = load_dataset("hotpot_qa", 'fullwiki', split='train')
        ds = ds[len(ds)//2:]
        ds = datasets.Dataset.from_dict(ds)
        parts = int(0.2*len(ds))
        ds = ds[:parts]
        ds = datasets.Dataset.from_dict(ds)
    elif data == 'triviaqa':
        ds = load_dataset('mandarjoshi/trivia_qa','rc', split='train')
        ds = ds[len(ds)//2:]
        ds = datasets.Dataset.from_dict(ds)
        parts = int(0.2*len(ds))
        ds = ds[:parts]
        ds = datasets.Dataset.from_dict(ds)
elif method == 'original-wts-stage2':
    if original_wts_type == 'cascade':
        path = './data/original_cascade/{}/{}_{}_{}.jsonl'.format(data,model,teacher, data)
    else:
        path = './data/original/{}_{}_{}_{}teacher.jsonl'.format(model, teacher,data,original_wts_type)
    with open(path) as f:
        ds = [json.loads(line) for line in f]
        ds = datasets.Dataset.from_list(ds)
elif method == 'ours-wts-stage1' or method == 'original-wts-stage1':
    if data == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", split='train')
        ds = ds[:len(ds)//2]
        ds = datasets.Dataset.from_dict(ds)
    else:
        with open('./data/CoT/{}_{}_{}_cot_correct.jsonl'.format(model,teacher,data)) as f:
            ds = [json.loads(line) for line in f]
            ds = datasets.Dataset.from_list(ds)            
elif method == 'ours-wts-stage2':
    if ours_wts_type == 'singleturn':
        with open('./data/ours/{}/{}_{}_to_{}_{}.jsonl'.format(data,model,teacher, student, data)) as f:
            ds = [json.loads(line) for line in f]
            ds = datasets.Dataset.from_list(ds)
    elif ours_wts_type == 'cascade1':
        with open('./data/ours/{}/{}_{}_to_{}_{}.jsonl'.format(data,model,teacher, student, data)) as f:
            ds = [json.loads(line) for line in f]
            ds = ds[:len(ds)//2]
            ds = datasets.Dataset.from_list(ds)
    elif ours_wts_type == 'cascade2':
        with open('./data/ours_cascade_halfhalf/{}/{}_{}_to_{}_{}.jsonl'.format(data,model,teacher, student, data)) as f:
            ds = [json.loads(line) for line in f]
            ds = datasets.Dataset.from_list(ds)
    else:
        with open('./data/ours_{}/{}/{}_{}_to_{}_{}.jsonl'.format (ours_wts_type,data,model,teacher, student, data)) as f:
            ds = [json.loads(line) for line in f]
            ds = datasets.Dataset.from_list(ds)
    
   
if model == 'qwen2.5':
    if method == 'ours-wts-stage1' or method == 'original-wts-stage1':
        model_id = "Qwen/Qwen2.5-{}-Instruct".format(teacher)
    else:
        model_id = "Qwen/Qwen2.5-{}-Instruct".format(student)
elif model == 'llama3':
    if method == 'ours-wts-stage1' or method == 'original-wts-stage1':
        if teacher == '1B' or teacher == '3B':
            model_id = "meta-llama/Llama-3.2-{}-Instruct".format(teacher)
        else:
            model_id = "meta-llama/Llama-3.1-{}-Instruct".format(teacher)
    else:
        if student == '1B' or student == '3B':
            model_id = "meta-llama/Llama-3.2-{}-Instruct".format(student)
        else:
            model_id = "meta-llama/Llama-3.1-{}-Instruct".format(student)

HF_token = 'xxxxxxxx'
print('Model trained:', model_id)

wandb.login()
wandb.init(
    project='weak-to-strong',
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
md = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token = HF_token
)

if model == 'llama3':
    tokenizer.add_special_tokens({"pad_token": "<unk>"})
    md.config.pad_token_id = tokenizer.pad_token_id


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    idx = random.random()
    for item in batch:
        if method == 'original-wts-stage1' and original_wts_type == 'qa':
            if data == 'hotpotqa' or data == 'triviaqa' or data == 'arc_challenge':
                instruction = item['question']
                response = item['ground_truth_response']
            elif data == 'gsm8k':
                instruction = item['question']
                response = '#### ' + item['answer'].split('####')[1].strip()
        elif method == 'base' and data == 'triviaqa':
            instruction = item['question']
            response = item['answer']['value']
        else:
            instruction = item['question']
            response = item['answer']
        if model == 'qwen2.5':
            instruction_temp = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n'
            response_temp = '<|im_start|>assistant\n{}<|im_end|>'
        elif model == 'llama3':
            instruction_temp = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 05 Jan 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>'
            response_temp = '<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>'
        formatted_instruction = tokenizer(instruction_temp.format(instruction), padding = True, max_length = 2048, truncation = True)
        formatted_response = tokenizer(response_temp.format(response), padding = True, max_length = 2048, truncation = True)

        input_id = torch.tensor(formatted_instruction['input_ids'] + formatted_response['input_ids'])

        attention = torch.tensor(formatted_instruction['attention_mask'] + formatted_response['attention_mask'])
        instruction_len = len(formatted_instruction['input_ids'])
        label = torch.tensor([-100] * instruction_len + formatted_response['input_ids'])
        
        input_ids.append(input_id)
        attention_mask.append(attention)
        labels.append(label)
        idx += 1
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    labels = torch.LongTensor(labels)
    
    out = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'labels': labels.to(device)
    }
    
    return out


training_dataloader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)


epochs = 3
learning_rate = 1e-5
accu_grad_steps = 16
weight_decay = 0.01
device = accelerator.device

if method == 'base':
    output_dir = "./model_outputs/base/{}_{}_{}".format(model,student,data)
elif method == 'original-wts-stage1' and original_wts_type == 'qa':
    output_dir = "./model_outputs/original/{}_{}_{}".format(model,teacher,data)
elif method == 'original-wts-stage2':
    if original_wts_type == 'cascade':
        output_dir = "./model_outputs/original_cascade/{}_{}_to_{}_{}".format(model,teacher, student, data, original_wts_type)
    else:
        output_dir = "./model_outputs/original/{}_{}_to_{}_{}_{}teacher".format(model,teacher, student, data, original_wts_type)
elif method == 'ours-wts-stage1' and ours_wts_type == 'singleturn':
    output_dir = "./model_outputs/ours/{}_{}_{}".format(model,teacher, data)
elif method == 'ours-wts-stage2':
    if ours_wts_type == 'singleturn':
        output_dir = "./model_outputs/ours/{}_{}_to_{}_{}".format(model,teacher, student, data)
    elif ours_wts_type == 'cascade1' or ours_wts_type == 'cascade2':
        output_dir = "./model_outputs/ours_cascade_halfhalf/{}_{}_to_{}_{}".format(model,teacher, student, data)
    else:
        output_dir = "./model_outputs/ours_{}/{}_{}_to_{}_{}".format(ours_wts_type,model,teacher, student, data)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, md.parameters()), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler =  get_cosine_schedule_with_warmup(optimizer, 100,
                                                epochs * len(training_dataloader) // accu_grad_steps)

md, optimizer, training_dataloader, scheduler = accelerator.prepare(md, optimizer, training_dataloader, lr_scheduler)


def train_one_epoch():
    for i, batch in enumerate(tqdm(training_dataloader)):
        with accelerator.accumulate(md):
            optimizer.zero_grad()
            #   inputs, targets = batch
            #   inputs = inputs.to(device)
            #   targets = targets.to(device)
            outputs = md(**batch)
            loss = outputs.loss
            # loss = loss/accu_grad_steps
            accelerator.backward(loss)
            
            wandb.log({"Train Loss": loss})
        
        # if (i+1) % accu_grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        # if (i+1) % 3000 == 0:
        #     if os.path.exists(output_dir):
        #         pass
        #     else:
        #         os.makedirs(output_dir, exist_ok=True)
        #     torch.save(model.state_dict(), os.path.join(output_dir, f"model_{i+1}.pt"))


for epoch in range(epochs):
    print("Epoch: {}, Training ...".format(epoch+1))
    md.train(True)
    train_one_epoch()
    if os.path.exists(output_dir):
        pass
    else:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(md.state_dict(), os.path.join(output_dir, f"model_{epoch+1}epoch.pt"))
    torch.cuda.empty_cache()
    


if os.path.exists(output_dir):
    pass
else:
    os.makedirs(output_dir, exist_ok=True)
torch.save(md.state_dict(), os.path.join(output_dir, "model_final.pt"))
