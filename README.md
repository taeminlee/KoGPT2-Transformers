# KoGPT2-Transformers

KoGPT2 on Huggingface Transformers

### KoGPT2-Transformers

- [SKT-AI 에서 공개한 KoGPT2(1.0)](https://github.com/SKT-AI/KoGPT2)를 [Transformers](https://github.com/huggingface/transformers)에서 사용하도록 하였습니다.

- ** SKT-AI 에서 KoGPT2 2.0을 공개하였습니다!! 💕 huggingface transformers가 first-order입니다! 👍 https://huggingface.co/skt/kogpt2-base-v2/ **

### Demo

- 일상 대화 챗봇 : http://demo.tmkor.com:36200/dialo
- 화장품 리뷰 생성 : http://demo.tmkor.com:36200/ctrl

### Update

#### 0.4.0

- transformers 4.0의 변화된 API에 대응하도록 변경하였습니다.
- 설치 없이 사용 가능! 😘
- 3.x API 사용 시 0.3.x 버전을 사용 바랍니다.

#### 0.3.1

- kogpt2_transformers.get_kogpt2_tokenizer() 사용 시 special_token_dict 에 `<unused0>` ~ `<unused97>` 토큰 추가

#### 0.3.0

- transformers 3.0의 변화된 API에 대응하도록 변경하였습니다.
- 2.x API 사용 경우 0.2.0 버전을 사용 바랍니다.

#### 0.2.0

- huggingface의 tokenize 패키지를 이용하도록 변경하였습니다. 속도 개선!

#### 0.1.0

- 최초 릴리즈

### Requirements

- transformers >= 4.0.0
- tokenizers >= 0.10.0
- torch >= 1.1.0

### Installation (option)

- `pip install kogpt2-transformers`

### Example 

- using pip package

```python
import torch
from kogpt2_transformers import get_kogpt2_model, get_kogpt2_tokenizer

torch.manual_seed(42)

model = get_kogpt2_model()
tokenizer = get_kogpt2_tokenizer()

input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
```

output

```sh
GENERATED SEQUENCE : 안녕히 자라</s><s> 오빠 어디야?</s><s> 걱정되게.</s><s> 어디야?</s><s> 연락이 안 되네...</s><s> 전화해 꼭</s><s> 내가 전화 꺼 놓을 꺼야?</s><s> 그래 잘 자</s><s> 나 이제 집에 가.</s><s> 잘 자,,,,</s><s> 나 이제 집에 가요</s><s> 전화 꼭 받으세요 기다릴께요 기다릴께요</s><s> 나 이제 잘려구...</s><s> 오빠두 잘래...</s><s> 잘 자</s><s> 너
GENERATED SEQUENCE : 안녕한 밤에 안녕</s><s> 야 저나해</s><s> 나 미지</s><s> 안녕하세요</s><s> 미래캐피탈입니다.</s><s> 최저 연 7</s><s> 누구나 100</s><s> 5000 만까지 당일 송금.</s><s> 연체 자 가능</s><s> 뭐 하시오?</s><s> 나 짐 대전 출발함</s><s> 낼 볼 일 있으시면 들리셔서 차 한 잔 하시며, 차 한 잔 하시삼</s><s> 보고파서.</s><s> 모 해?</s><s> 너가 어제 문자 보냈던 그
GENERATED SEQUENCE : 안녕!</s><s> 너는 이제부터 다시 너에게 의지할 꺼야.</s><s> 난 정말 너를 사랑하고 잇어.</s><s> 너 때문에 많이 아파해서 죽고 싶진 않을 꺼야.</s><s> 정말 너무 힘들다.</s><s> 너의 맘 변하지 않도록 기도할께.</s><s> 사랑해요.</s><s> 젼</s><s> 정말이지 널 믿엇던 약속이 거짓말인 줄 알면서도 더 이상 너에게 의지하지 않을께.</s><s> 정말 너무 힘들어서 살기 어렵다 정말
```

- using transformers classes

```python
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

torch.manual_seed(42)

model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")

input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
```
