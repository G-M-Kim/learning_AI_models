# 실험 규칙과 실제 CPU 예시

아래는 실행으로 얻은 값입니다. RTX 4070 Ti 결과가 아닙니다. Python 3.12 / PyTorch 2.7.1+cpu / numpy 2.2.6 / scikit-learn 1.6.1, seed 42, Adam lr 0.001, batch 64, 15 epochs. GPU 설치 안내는 Windows Python 3.11 + CUDA wheel을 기준으로 별도 제공했습니다.

CPU 참고 실험은 일부 동시 실행했으므로 시간은 성능 비교 근거로 쓰지 않습니다. 같은 task/길이 안에서 accuracy와 loss를 관찰하세요. seed 1개이므로 우열을 확정할 수 없습니다. `examples/cpu_reference/`에 결과·설정·loss history 원본을 포함했고 checkpoint는 포함하지 않았습니다. `comparison.html`을 브라우저로 열면 곡선을 볼 수 있습니다.

| 과제 | 길이 | 모델 | Test accuracy | Test loss | PPL |
|---|---:|---|---:|---:|---:|
| digits | 8×8 image | cnn | 95.19% | 0.1901 | — |
| digits | 8×8 image | dnn | 91.85% | 0.2583 | — |
| digits | 8×8 image | scnn | 87.78% | 0.8679 | — |
| digits | 8×8 image | snn | 93.33% | 0.3413 | — |
| language | 32 | gpt | 67.15% | 0.5649 | 1.7593 |
| language | 32 | lstm | 67.24% | 0.5658 | 1.7609 |
| recall | 8 | dnn | 100.00% | 0.0045 | — |
| recall | 8 | gru | 100.00% | 0.0044 | — |
| recall | 8 | lstm | 100.00% | 0.0050 | — |
| recall | 8 | rnn | 100.00% | 0.0053 | — |
| recall | 8 | spike_attention | 100.00% | 0.0028 | — |
| recall | 8 | srnn | 99.90% | 0.0153 | — |
| recall | 8 | transformer | 100.00% | 0.0010 | — |
| recall | 32 | dnn | 100.00% | 0.0075 | — |
| recall | 32 | gru | 12.99% | 2.0785 | — |
| recall | 32 | lstm | 11.82% | 2.0791 | — |
| recall | 32 | rnn | 100.00% | 0.0183 | — |
| recall | 32 | spike_attention | 100.00% | 0.0031 | — |
| recall | 32 | srnn | 26.17% | 1.9223 | — |
| recall | 32 | transformer | 100.00% | 0.0013 | — |

## 여기서 배울 수 있는 점

- Digits CNN은 DNN보다 +3.33 percentage points였습니다. 이 작은 설정에서의 관찰이며 CNN 일반 성능 보장은 아닙니다.
- Recall 길이 8에서는 LSTM/GRU도 100%에 도달했지만 32에서는 chance 근처에 머물렀습니다. Vanilla RNN은 이 seed에서 32도 풀었습니다. 더 강한 기억 구조가 같은 learning rate/epoch에서 자동으로 더 잘 최적화되는 것은 아닙니다. 이 결과를 숨기거나 test를 보며 하이퍼파라미터를 고쳐서 "발전 순서"에 맞추지 않았습니다.
- Recall DNN은 첫 위치에서 출력까지 바로 연결되어 쉽습니다. 과제의 구조에 맞는 경로가 있는지가 모델의 시대보다 중요합니다.
- 언어 모델의 PPL은 LSTM과 GPT가 거의 같았습니다. 단순 문법의 짧은 문장에서는 Transformer의 대규모 모델 이점을 볼 수 없습니다. 확률적으로 선택한 색/동물 때문에 정답을 항상 하나로 예측할 수는 없습니다.
- SNN/SCNN/SRNN은 accuracy 최적화를 위한 완성 모델이 아니라 입력 표현·막전압·surrogate gradient를 배우는 출발점입니다.

## 사용자 GPU에서 비교하는 절차

1. `python -m ailab.doctor --require-cuda`로 forward/backward 확인.
2. `python -m ailab.suite --task digits --epochs 15 --seeds 42 43 44`.
3. recall은 길이 8과 32를 별도로 suite 실행.
4. language는 동일 epochs/seeds로 suite 실행.
5. `python -m ailab.report` 후 `runs/comparison.html`에서 동일 group 내 비교.

## 한 번에 바꿀 변수 하나

| 질문 | 바꿀 것 | 고정할 것 |
|---|---|---|
| 긴 기억이 어려운가? | recall `--length 8/16/32/64` | 모델, epoch, lr, seed 목록 |
| 학습이 덜 되었나? | `--epochs 15/30` | 데이터와 나머지 설정 |
| spike 시간 해상도는? | ImageSNN.steps 코드 값 | 데이터, neuron rule, 별도 코드 hash 기록 |
| 초기화에 민감한가? | `--seeds 42 43 44` | 모든 학습 설정, split |

동일 epoch는 동일 데이터 노출 예산입니다. 파라미터 수·FLOPs·훈련 시간을 일치시킨 실험은 아닙니다. 모델마다 더 좋은 learning rate가 있을 수 있습니다. 조정은 validation으로 수행하고, 최종 선택 이후 test를 봅니다.

## 재현성 범위

데이터·주요 코드 SHA256·주요 package/장치 정보를 config에 저장합니다. 전체 pip 환경, 드라이버, GPU kernel의 비결정성까지 완전히 고정하지는 않습니다. 서로 다른 장치/버전에서 숫자가 완벽히 같아야 하는 실습은 아닙니다. 기록된 source hash는 참고 실험 당시의 data.py/models.py/run.py를 나타냅니다.
