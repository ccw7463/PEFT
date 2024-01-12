import os
import json
from token import OP
import torch
import transformers
from dataclasses import dataclass, field

from typing import Dict, Optional, Sequence

def jload(path) -> Dict:
    '''json load 함수'''
    with open(path,'r') as f:
        data = json.load(f)
    return data

@dataclass
class ModelArguments:
    '''
    Model 관련 args 설정
    
    아래 변수들은 train.sh (shell script) 에서 직접적으로 전달받는 인자
    
    전달받지 못할경우 default 값을 사용
    '''
    model_name_or_path : Optional[str] = field(default="EleutherAI/polyglot-ko-1.3b")
    tokenizer_name_or_path : Optional[str] = field(default="EleutherAI/polyglot-ko-1.3b")
    override_custom_token : Optional[bool] = field(default=True)
    use_flash_attention : Optional[bool] = field(default=True)
    
@dataclass
class DataArguments:
    '''
    Data 관련 args 설정 
    
    아래 변수들은 train.sh (shell script) 에서 직접적으로 전달받는 인자
    
    전달받지 못할경우 default 값을 사용
    '''
    data_path : str = field(default=None)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    '''
    학습 관련 args 설정 
    
    아래 변수들은 train.sh (shell script) 에서 직접적으로 전달받는 인자
    
    전달받지 못할경우 default 값을 사용
    '''
    cache_dir : Optional[str] = field(default="./models")
    optim : str = field(default="adamw_torch")
    model_max_length : int = field(default=512)
