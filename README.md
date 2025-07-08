# [COLING'25] RRHF-V: Ranking Responses to Mitigate Hallucinations in Multimodal Large Language Models with Human Feedback


# Rank-responses Construction Pipeline

skip


# Environment Setup

```
conda create -yn rrhf-v python=3.9
conda activate rrhf-v
cd RRHF-V
pip install -r requirements.txt
```


# Train

llava-1.5-7b
```
python3 RRHF-V_llava-v1_5-7b-hf.py
```

tiny-llava-1b
```
bash RRHF-V_llava-v1_5-7b-hf.py
```