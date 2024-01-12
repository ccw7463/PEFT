# PEFT(Parameter-Efficient Fine-Tuning)
This is a repository for organizing my studies on PEFT
💡
전체 파라미터를 Fine-Tuning 하는게 아닌 필요한 파라미터만 재학습하는 비용효율적 방법
1. PEFT 특징
💡
PEFT 이해
대규모 모델의 파라미터를 대부분 고정한 채로 일부 파라미터만을 학습한다. 일부만 재학습하므로 GPU 메모리 요구량이 낮으며, Foundation Model의 언어능력과 사전학습에 사용된 데이터에 대한 정보가 대부분 보존된다는 장점이 있음.



PEFT 장점

메모리 효율성
저장공간 최소화 (PEFT를 통해 학습된 모델 버전은 크기가 작음)
적응성과 유연성이 좋음 (다양한 Task에 적용하기 용이함)
예) 각기 다른 사용자에게 맞춤화된 챗봇 제작시 소수의 파라미터만 조정
재난적 망각(catastrophic forgetting) 방지
catastrophic forgetting 이란 새로운 데이터를 학습하면서 이전에 학습했던 정보를 대량으로 잃는것을 의미함


2. PEFT 종류와 특징
PEFT 기법에는 크게 Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) 가 있음. 이름에서도 알 수 있듯이, QLoRA의 경우 LoRA 방식을 Quantization 한 방식이다.



Low-Rank Adaptation (LoRA) 

💡
기존에 학습해야 하는 파라미터와 동일한 Shape를 갖도록 작은 행렬 2개를 추가하는 방식


Vanilla Transformer 를 예로 이해하기

Transformer Layer 1개의 경우 512 x 64 (32,768개)의 학습 파라미터를 가짐. 여기에 작은 행렬 A, B를 추가한다. (행렬곱 결과가 동일한 shape를 가지도록)


아래에서 ‘4’를 Rank라 칭함.
A 크기 : 512 x 4  
B 크기 : 4 x 64 
Rank 값은 4~16 에서 좋은 성능을 보였음. (예 : 4,8,16)


따라서, Fine-Tuning 시

기존 512 x 64 shape를 가지는 파라미터 고정 
A, B 행렬을 학습 → (2048+256)개의 파라미터만 학습하면 됨.