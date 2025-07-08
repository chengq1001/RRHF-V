# [COLING'25] RRHF-V: Ranking Responses to Mitigate Hallucinations in Multimodal Large Language Models with Human Feedback

[pdf](https://aclanthology.org/2025.coling-main.454/)


![pdf preview](./assets/intro-rrhf-v-7.pdf)


## Abstract
Multimodal large language models (MLLMs) demonstrate strong capabilities in multimodal understanding, reasoning, and interaction but still face the fundamental limitation of hallucinations, where they generate erroneous or fabricated information. To mitigate hallucinations, existing methods annotate pair-responses (one non-hallucination vs one hallucination) using manual methods or GPT-4V, and train alignment algorithms to improve the correspondence between images and text. More critically, an image description often involve multiple dimensions (e.g., object attributes, posture, and spatial relationships), making it challenging for the model to comprehensively learn multidimensional information from pair-responses. To this end, in this paper, we propose RRHFV, which is the first using rank-responses (one non-hallucination vs multiple ranking hallucinations) to mitigate multimodal hallucinations. Instead of using pair-responses to train the model, RRHF-V expands the number of hallucinatory responses, so that the responses with different scores in a rank-response enable the model to learn rich semantic information across various dimensions of the image. Further, we propose a scene graph-based approach to automatically construct rank-responses in a cost-effective and automatic manner. We also design a novel training objective based on rank loss and margin loss to balance the differences between hallucinatory responses within a rankresponse, thereby improving the model’s image comprehension. Experiments on two MLLMs of different sizes and four widely used benchmarks demonstrate that RRHF-V is effective in mitigating hallucinations and outperforms the DPO method based on pair-responses.


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