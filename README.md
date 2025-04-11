# Alice: Proactive Learning with Teacher's Demonstrations for Weak-to-Strong Generalization🐇

## 🪴Overview
We introduce Alice, a new wrak-to-strong generalization framework that fully utilizes both teacher and student models’ intelligence and enables them to complement each other’s knowledge, finally achieving superior generazalition outcome. Alice starts by probing the real knowledge base of weak teacher models through uncertainty expression. Students are then instructed to leverage both the teacher's task-specific guidance and its own superior capabilities to self-generate higher-quality responses, which serve as training supervisions later. For scenarios with substantial capability gaps between teacher and student models, we also introduce cascade Alice, a multi-stage iterative weak-to-strong supervision framework that uses weak teachers to guide intermediate models, which then serve as teachers for stronger models, enabling more stable knowledge transfer.


![Alice](./Alice.png)
## 🪵Requirements
1. The required Python packages for running this repo are listed in [requirements.txt](./requirements.txt). To install these pacakages at one time, plaese run
```shell
pip install -r requirements.txt
```

2. You also need to set up variable `HF_token`(your Huggingface access token) in most scripts.
3. Model `./Meta-Llama-3-70B-Instruct` is used in our weak-to-strong generalization pipeline, and you may need to download it from [Huggingface](https://huggingface.co/meta-llama/Meta-Llama-3-70B) and store it under folder `Alice` before getting started.
   
Note: We have preprocessed the `ARC_challenge` dataset for easier use later. The processed dataset has already been included in the folder at [arc_challenge_processed](.arc_challenge_processed).


## 🐾CoT Generation
As a step one, we'll need to generate CoT for each q-a pair for datasets that only comes with ground truth answer label: HotpotQA, TriviaQA, and ARC_challenge. The scripts are stored in folder [cot](./cot_generate_filter). You need to specify the following command line argument when running the scripts: 
1) `--data`: Choose from 'hotpotqa', 'triviaqa', and 'arc_challenge'.
2) `--model`: Choose from 'qwen2.5' and 'llama3'
3) `--teacher`: The size of the teacher model. Choose from '1.5B' and '3B' when you specify 'qwen2.5' for model, and choose from '1B' and '3B' when you do 'llama3'.

Here is an example. To generate the CoT reasoning content that matched the ground truth answer for dataset 'TriviaQA' of the teacher model 'Llama3 1B', you should run:
```shell
python cot/cot_generate_filter.py --model llama3 --data triviaqa --teacher 1B
```
The generated data will be stored in a new folder `./data/CoT` fter running the scripts. 


## 🧚🏻Training Teacher Models with CoT
Next, we'll use the first half of the dataset to train teacher models so that they are equipped with basic domain knowledge. You'll need to run the scripts in folder [training](./training/SFT.py). 
For our new method, we need to train teacher models with CoT. You need to specify the following command line argument when running the scripts: 
1) `--data`: Choose from 'hotpotqa', 'triviaqa', 'gsm8k' and 'arc_challenge'.
2) `--model`: Choose from 'qwen2.5' and 'llama3'
3) `--teacher`: The size of the teacher model. Choose from '1.5B' and '3B' when you specify 'qwen2.5' for model, and choose from '1B' and '3B' when you do 'llama3'.
4) `--method`: You should use 'ours-wts-stage1'
5) `--our-wts-type`: You should use 'singleturn'

For the original W2S method, we have two separate settings: train teachers with cot or with q-a pair only. For the cot one, it's the same as the stage 1 for our approach, so you just need to follow the instructions above. (You just need to run it once and the output models can be used for both settings). For the qa one, you should follow the instructions above for the first three arguments and made the following changes to the last two:
1) `--method`: You should use 'original-wts-stage1'
2) `--original-wts-type`: You should use 'qa'

After running the scripts, the model outputs will be stored in a new folder `./model_outputs`.

## 🪺Original W2SG

## 🍄New W2SG - Alice

## 🍃Baselines: Weak/Strong Performance
We train weak teacher/strong students using last half of ground truth label for each dataset. The evaluation results are taken as the weak/strong performance, which serve as the baselines in this work. You can simply run the [SFT.py](./training/SFT.py) scripts by setting the argument `--method` as 'base' and specifying the weak/strong model size in argument `--student`. For instance, if you want to get the weak teacher performance for qwen2.5 1.5B on gsm8k dataset, you may run:
```shell
python training/SFT.py --model qwen2.5 --data gsm8k --student 1.5B --method base
```

## 🐚Evaluation





