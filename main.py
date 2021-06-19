from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from transformers import GPT2LMHeadModel
from pydantic import BaseModel, Field

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>') 



model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

#input 정의 - BaseModel 상속
class Input(BaseModel):
    text: str = Field(
        title = '문장을 입력해주세요',
        max_length= 128 #입력하는 문장의 최대길이
    )
    max_length: int = Field( #생성하는 문장의 최대길이 
        128,
        ge = 5, # ge = greater than equal
        le = 128 # le = less than equal
    )   
    repetitaion_penalty: float = Field(
        2.0,
        ge = 0.0,
        ke = 2.0
    )

class Output(BaseModel):
    generated_text: str

def generate_text(input: Input) -> Output:  # 함수 : input은 Input으로 받는 
    input_ids = tokenizer.encode(input.text) # 입력받을 텍스트는 Input class에서의 text변수
    gen_ids = model.generate(torch.tensor([input_ids]),
        max_length=input.max_length,
        repetition_penalty=input.repetitaion_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True)
    generated = tokenizer.decode(gen_ids[0,:].tolist())
    
    return Output(generated_text=generated)