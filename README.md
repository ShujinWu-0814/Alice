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
2) `--model`: Choose from 'qwen2.5' and 'llama 3'
3) `--teacher`: The size of the teacher model. Choose from '1.5B' and '3B' when you specify 'qwen2.5' for model, and choose from '1B' and '3B' when you do 'llama3'.

Here is an example. To generate the CoT reasoning content that matched the ground truth answer for dataset 'TriviaQA' of the teacher model 'Llama3 1B', you should run:
```shell
python cot/cot_generate_filter.py --model llama3 --data triviaqa --teacher 1B
```
The generated data will be stored in a new folder `./data/CoT` fter running the scripts. 

