#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4 python W2SG_pipeline/ours/new_weak_to_strong_cluster.py --data hotpotqa --teacher 1.5B --midlevel 3B --student 7B --method cascade --model qwen2.5
CUDA_VISIBLE_DEVICES=1,2,3,4 python W2SG_pipeline/ours/new_weak_to_strong_uncertainty.py --data hotpotqa --teacher 1.5B --midlevel 3B --student 7B --method cascade --model qwen2.5
CUDA_VISIBLE_DEVICES=1,2,3,4 python W2SG_pipeline/ours/new_weak_to_strong_zeroshot.py --data hotpotqa --teacher 1.5B --midlevel 3B --student 7B --method cascade --model qwen2.5
CUDA_VISIBLE_DEVICES=1,2,3,4 python W2SG_pipeline/ours/new_weak_to_strong_final.py --data hotpotqa --teacher 1.5B --midlevel 3B --student 7B --method cascade --model qwen2.5
