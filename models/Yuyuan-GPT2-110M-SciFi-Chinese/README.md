---
language: 
  - zh

inference: 
  parameters:
    temperature: 1
    top_p: 0.7
    repetition_penalty: 1.1
    max_new_tokens: 128
    num_return_sequences: 3
    do_sample: true

license: apache-2.0
tags:
- generate
- gpt2

widget:
- 我是逻辑，面对黑暗森林法则，
- 天空的尽头，发出了一道亮眼的光芒，

---

# Yuyuan-GPT2-110M-SciFi-Chinese

- Main Page:[Fengshenbang](https://fengshenbang-lm.com/)
- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)

## 简介 Brief Introduction

基于中文版的Wenzhong-GPT2-110M，我们用接近2万的科幻小说数据进行了微调模型，让模型能够较好的续写科幻小说。

Based on the Wenzhong-GPT2-110M, we used nearly 20000 science fiction data to fine-tune the model, so that the model can better continue to write science fiction.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 科幻小说 science fiction  | 自然语言生成 NLG | 余元 Yuyuan | GPT2 |      110M      |     中文 Chinese     |

## 模型信息 Model Information

模型结构和wenzhong-110M相同，只是微调的语料由通用数据变成了科幻小说。

The structure of the model is the same as that of Wenzhong-GPT2-110M, except that the fine-tuned corpus has changed from general data to science fiction.

## 使用 Usage

### 加载模型 Loading Models

```python 
from transformers import GPT2Tokenizer,GPT2LMHeadModel
hf_model_path = 'IDEA-CCNL/Yuyuan-GPT2-110M-SciFi-Chinese'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
```

### 使用示例 Usage Examples

```python
question = "我是逻辑，面对黑暗森林法则，"
inputs = tokenizer(question,return_tensors='pt')
generation_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=150,
                                # max_new_tokens=80,
                                do_sample=True,
                                top_p = 0.6,
                                # num_beams=5,
                                eos_token_id=50256,
                                pad_token_id=0,
                                num_return_sequences = 5)

for idx,sentence in enumerate(generation_output.sequences):
    print('next sentence %d:\n'%idx,
    tokenizer.decode(sentence).split('<|endoftext|>')[0])
    print('*'*40)
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Jiaxing Zhang and Ruyi Gan and Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
