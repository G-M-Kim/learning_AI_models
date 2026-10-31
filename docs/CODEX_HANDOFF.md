# Codex 인수인계 · 다음 수정부터 쓰는 안내

## 핵심

기존 대화를 모두 복사할 필요는 없습니다. 최신 저장소를 열고, 이 문서와 루트의 `AGENTS.md`를 읽은 다음 이번 수정 요구와 완료 조건을 전달하면 됩니다. 이 문서는 현재 구현의 출발점을 설명하며, 이후에는 실제 코드·현재 README·사용자의 최신 지시를 우선합니다.

`AGENTS.md`는 에이전트가 저장소에서 따라야 할 작업 지침입니다. `CODEX_HANDOFF.md`는 사람과 에이전트가 함께 읽는 맥락·파일 안내·요청 예시입니다. 별도의 설치형 skill은 아닙니다.

## 현재 프로젝트

- 저장소: https://github.com/G-M-Kim/learning_AI_models
- 목적: 각 모델의 코드를 읽고 실제 학습한 뒤 동일한 test set에서 성능/비용 차이를 관찰.
- 사용자 환경: Windows 11 / RTX 4070 Ti / DDR5 32 GB.
- 설치 기준: Python 3.11 / torch 2.7.1 CUDA 12.6 / requirements.txt.
- 외부 API 키 불필요. 데이터는 sklearn 내장 digits 또는 고정 seed 생성기.
- 시각화 교재: `docs/study.html` (브라우저로 열기).
- 설치·다운로드·업데이트: `docs/SETUP_WINDOWS.md`.
- 기본 지표: accuracy/loss, params, 시간, CUDA tensor allocation. LM은 PPL 추가.
- 비교 단위: 동일 task/data/protocol group. 모델·seed만 비교 축으로 다르게 함.

### 구현 목록

| task | models | 입력 → 정답 |
|---|---|---|
| digits | dnn, cnn, snn, scnn | [B,1,8,8] → [B] 10분류 |
| recall | dnn, rnn, lstm, gru, srnn, transformer, spike_attention | [B,T,10] one-hot → [B] 첫 기호 8분류 |
| language | lstm, gpt | [B,T] token IDs → [B,T] 다음 token |

Recall의 기호 0~7이 일반 기호이고, 마지막 기호 8은 query입니다. feature 9는 예약된 미사용 차원입니다. DNN은 첫 위치를 직접 볼 수 있어 recall을 쉽게 풀 수 있습니다. 이 과제는 Transformer가 반드시 MLP보다 우월함을 증명하기 위한 것이 아닙니다.

### 파일 지도

| 파일 | 역할 |
|---|---|
| `ailab/data.py` | split, 입력/정답 생성, vocabulary, 데이터 fingerprint |
| `ailab/models.py` | 모델·LIF cell·surrogate gradient·causal mask |
| `ailab/run.py` | 학습 → validation best checkpoint → test 한 번 |
| `ailab/suite.py` | 모델·seed 순차 실행 |
| `ailab/predict.py` | checkpoint로 test 예측/문장 생성 |
| `ailab/doctor.py` | CUDA 환경과 forward/backward 확인 |
| `ailab/report.py` | 결과 그룹화, seed 평균/SD, 비교 HTML·CSV |
| `scripts/build_study.py` | 교재 생성 소스; 수정 후 실행하여 HTML 갱신 |
| `tests/test_lab.py` | shape/gradient, 미래 누출, SNN reset, split, metric 검사 |
| `.github/workflows/check.yml` | Windows/Linux CPU 검사 |
| `examples/cpu_reference/` | 초기 20회 CPU 실측; GPU 성능 기준이 아님 |

## 검증 상태와 남은 한계

초기 버전은 Linux CPU에서 모든 모델 조합의 학습과 checkpoint 재사용을 실행했습니다. 20회 full reference run은 seed 42, 15 epochs이며, 일부를 동시 실행했으므로 CPU wall time은 모델 속도 순위의 근거로 사용하지 않습니다. 단위 테스트 6개가 모든 모델 조합의 backward 등을 검사합니다.

사용자 RTX 4070 Ti에서의 실제 실행은 아직 확인 전입니다. Windows/Linux CI의 최신 상태는 GitHub의 실제 Checks를 확인해야 합니다. 과거에 통과했다고 추정하지 마세요.

학습 교재의 HTML 구조·내부 링크와 JavaScript 구문은 검사했지만, 최초 제작 환경에서 Chromium 설치가 실패하여 실제 브라우저 렌더링 QA는 수행하지 못했습니다. HTML 시각 요소를 수정한다면 가능한 환경에서 브라우저로 확인하세요.

모델 폭·depth·FLOPs가 동일하지 않은 작은 교육용 비교입니다. spike_attention은 analog embedding/residual/readout을 포함한 단일 step hybrid이며 논문 재현이나 spiking LLM이 아닙니다. pretrained 모델 fine-tuning, BERT MLM, Seq2Seq 번역은 아직 포함하지 않았습니다.

## VS Code의 Codex에게 요청하는 순서

1. 작업 중인 코드가 있으면 먼저 별도 branch에 commit하거나 백업합니다.
2. main에 올라온 새 코드를 쓸 때만 `git switch main`, `git pull --ff-only origin main`을 실행합니다. 미완료 PR에서 계속 작업할 때는 그 PR branch를 사용합니다.
3. VS Code에서 **저장소 폴더 자체**를 엽니다. Codex 확장을 설치·로그인하고 Codex 패널을 엽니다.
4. 다음 기본 요청에 이번 요구를 붙입니다.

```text
이 저장소는 AI 모델 발전을 순서대로 배우는 toy project입니다.
먼저 AGENTS.md, docs/CODEX_HANDOFF.md, README.md와 관련 코드를 읽어주세요.
Windows 11 / RTX 4070 Ti / Python 3.11 환경이며,
학습용으로 읽기 쉬운 코드와 한국어 교재를 유지하려 합니다.

이번 수정:
- [원하는 기능 또는 고칠 문제]

완료 조건:
- [어떤 명령/입력에서 어떤 결과가 나오면 되는지]
- 관련 기존 모델과 같은 데이터/프로토콜로 비교할 수 있을 것.
- 필요한 검증을 실행하고 교재/README도 구현에 맞게 갱신할 것.

기존 개인 runs와 관련 없는 로컬 수정은 보존해주세요.
결과에 변경 내용, 검사 결과, GPU 검증 여부,
branch/commit 및 제가 다음에 실행할 명령을 알려주세요.
```

## 요청 예시

### 기능 추가

```text
Recall 실험 결과에서 길이별 test accuracy를 한 그래프에 보고 싶습니다.
기존 비교 group 규칙은 유지하면서,
길이만 달리한 결과를 명시적으로 선택해 비교하는 기능을 추가해주세요.
데이터/epoch/lr/seed 차이를 그래프에서 숨기지 않도록 해주세요.
```

### 실행 오류 해결

```text
아래 명령을 실행했더니 오류가 납니다.
실행 명령: [복사]
오류 전체: [Traceback 시작부터 마지막 줄까지 복사]
python -m ailab.doctor --require-cuda 출력: [복사]

원인을 찾아 최소한으로 고치고,
같은 오류가 재발하지 않는지 확인한 명령과 결과를 알려주세요.
```

### 코드 설명만 요청

```text
ailab/models.py의 SRNN.forward를 RTL 상태 레지스터와 조합논리 관점에
연결해서 설명해주세요. timestep마다 membrane/spike의 shape와
갱신 순서를 알려주세요. 이번에는 코드 변경은 하지 마세요.
```

## ChatGPT에서 논의한 변경을 Codex로 넘기기

ChatGPT에 먼저 "이 변경을 Codex에 전달할 작업 지시문으로 정리해줘. 목적, 수정 범위, 완료 조건, 검증 명령을 포함해줘"라고 요청하면 됩니다. 그 지시문을 저장소를 연 Codex에 붙여 넣으세요. 이후 결과는 commit/PR을 기준으로 여기서 다시 검토할 수 있습니다.

로컬 Codex는 현재 PC의 파일·환경을 다룰 수 있지만, 클라우드 실행이 사용자 RTX 4070 Ti를 자동으로 쓰는 것은 아닙니다. GPU 문제를 해결할 때는 실제 실행 위치와 doctor 출력을 함께 전달하세요.

공식 안내:
- https://learn.chatgpt.com/docs/ide
- https://learn.chatgpt.com/docs/projects
- https://learn.chatgpt.com/docs/agent-configuration/agents-md
