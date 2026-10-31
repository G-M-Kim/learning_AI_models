# Windows 11 · 처음 설치부터 RTX 4070 Ti 학습까지

이 프로젝트는 로컬 PyTorch 학습입니다. ChatGPT API 키, 유료 API 호출, CUDA C++ 코딩, Codex 설치는 실행에 필요하지 않습니다. 약 12 GB VRAM의 RTX 4070 Ti에서 사용하도록 작은 FP32 모델·batch 64로 구성했습니다. 여기서 제공하는 CPU 검증은 실제 사용자 GPU에서의 검증을 대신하지 않습니다.

## 1. 프로그램 설치 (한 번만)

1. NVIDIA 공식 사이트에서 RTX 4070 Ti용 최신 드라이버 설치 후 재부팅: https://www.nvidia.com/Download/index.aspx
2. Git for Windows 설치: https://git-scm.com/downloads/win (기본 옵션으로 진행).
3. Python **3.11 64-bit** 설치: https://www.python.org/downloads/windows/ 에서 3.11 계열 Windows installer (64-bit). 예: 3.11.9. 설치 화면의 **Add python.exe to PATH**, Python Launcher를 선택합니다. 새 버전 자동 선택 대신 이 실습의 기준인 3.11을 사용하세요.
4. VS Code 확장 `Python` 및 `Pylance` (게시자 Microsoft) 설치.
5. 열려 있던 VS Code를 모두 종료한 뒤 재실행합니다. Terminal → New Terminal에서 PowerShell을 엽니다.

```powershell
git --version
py -3.11 --version
nvidia-smi
```

세 명령이 모두 작동해야 합니다. `nvidia-smi`의 CUDA Version은 드라이버가 지원하는 상한이며 설치된 PyTorch 버전과 숫자가 같을 필요는 없습니다.

## 2. PR 반영과 최초 다운로드

ChatGPT가 안내한 PR 링크 → **Files changed**로 파일 확인 → **Checks** 성공 확인 → **Merge pull request** 또는 **Squash and merge** → **Confirm**. PR에 대한 `Approve` 리뷰만으로 main에 파일이 들어가지는 않습니다. 이 문서에는 존재하지 않는 PR 번호를 미리 적지 않았습니다.

VS Code PowerShell에서 프로젝트를 둘 위치로 이동한 후:

```powershell
cd $env:USERPROFILE
mkdir ai-study -Force
cd ai-study
git clone https://github.com/G-M-Kim/learning_AI_models.git
cd learning_AI_models
code .
```

새 창의 Terminal → New Terminal. 경로가 `learning_AI_models`로 끝나는지 확인하세요. 이미 clone했다면 다시 clone하지 말고 아래 '업데이트' 절을 사용합니다. PR을 merge하기 전에는 안내받은 실제 branch를 `git fetch origin` 및 `git switch --track origin/브랜치이름`으로 받아 볼 수도 있습니다.

## 3. 독립 Python 환경 및 GPU 라이브러리

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ailab.doctor --require-cuda
```

PyTorch 2.7.1 + CUDA 12.6은 최신 자동 추종 대신 선택한 실습 기준입니다. 공식 설치 명령: https://pytorch.org/get-started/previous-versions/#v271. NVIDIA 드라이버는 이 wheel을 실행할 수 있어야 합니다. 별도의 CUDA Toolkit/cuDNN 설치는 이 사전 빌드 wheel 실습에 필요하지 않습니다.

정상 출력: `CUDA available: True`, `GPU: NVIDIA GeForce RTX 4070 Ti`, `CUDA forward/backward: PASS`.

VS Code `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`. 목록에 없으면 Enter interpreter path로 선택합니다. 이 문서의 명령은 venv 활성화나 PowerShell 정책 변경 없이 실행됩니다. 아래의 짧은 `python` 명령을 쓰고 싶으면 터미널 세션마다 다음을 한 번 실행합니다:

```powershell
Set-Alias python "$PWD\.venv\Scripts\python.exe"
python -m ailab.doctor --require-cuda
```

**이후 README 및 교재의 `python`은 이 venv Python을 뜻합니다.** 새 터미널에서는 alias를 다시 설정하거나 `.\.venv\Scripts\python.exe`를 사용하세요.

## 4. 첫 실행과 결과 열기

```powershell
python -m ailab.run --task digits --model dnn --quick --epochs 2
python -m ailab.report
Start-Process .\runs\comparison.html
Start-Process .\docs\study.html
```

`--quick`은 split당 128개만 쓰는 기능 점검입니다. 여기서 얻은 정확도를 모델 성능 결론에 쓰지 않습니다. 정상 실행 후 전체 digits 비교:

```powershell
python -m ailab.suite --task digits --epochs 15 --seeds 42 43 44
```

한 번에 모델 하나만 GPU에 올립니다. 느리면 seed 42 하나로 시작하세요. 정확한 소요 시간은 실행 후 기록됩니다. 작은 모델이라 GPU 사용률이 낮을 수 있습니다.

## 5. 테스트 입력과 학습한 모델 재사용

학습 완료 시 `Saved: runs/...` 경로가 표시됩니다. PowerShell로 최신 체크포인트를 고를 수 있습니다:

```powershell
$Checkpoint = (Get-ChildItem .\runs -Recurse -Filter best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python -m ailab.predict --checkpoint "$Checkpoint"
```

digits/recall은 보관한 test sample의 정답과 예측을 출력합니다. language는 `<bos>`에서 시작해 60 token을 생성합니다. 임의의 손그림 PNG 입력은 현재 CLI 범위에 포함하지 않았습니다. 사용자가 직접 입력을 바꾸려면 `ailab/data.py`의 `recall()` 또는 `language()`를 읽고, 데이터가 바뀌면 새 비교 실험으로 취급하세요.

## 6. 이후 업데이트

```powershell
git status
git switch main
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

로컬에서 수정한 코드가 있으면 먼저 별도 branch에 commit하거나 백업합니다. `runs/`는 git에서 제외되므로 개인 학습 결과는 업로드되지 않습니다. `git reset --hard`를 업데이트 명령으로 사용하지 마세요.

## 7. 흔한 오류

| 증상 | 해결 |
|---|---|
| `py`/`git` 명령을 못 찾음 | 설치 확인 후 VS Code 재시작. Python Launcher와 PATH 옵션 확인 |
| `No module named torch` | 전역 Python이 아닌 `.\.venv\Scripts\python.exe`로 실행 |
| `CUDA available: False` | `nvidia-smi` 확인. `python -m pip show torch`로 환경 확인 후 아래 재설치 |
| CPU wheel을 잘못 설치함 | `python -m pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126` |
| 드라이버가 오래되었다는 오류 | NVIDIA 드라이버 업데이트 후 재부팅 |
| `CUDA out of memory` | 다른 GPU 앱 종료; `--batch-size 32`. 바뀐 batch끼리는 별도 비교 group |
| `No module named ailab` | 터미널 현재 폴더가 repository 루트인지 확인 |
| `Activate.ps1` 실행 불가 | 활성화가 필요 없는 위 명령 그대로 사용 |
| 그래프가 비어 있음 | 학습 완료 후 `python -m ailab.report` 재실행 |

## 8. Codex는 선택 사항

이 실습을 실행하는 데 Codex는 필요하지 않습니다. 코드 설명이나 PC에서 발생한 오류 수정을 함께 받고 싶으면 VS Code 확장에서 OpenAI의 **Codex**를 설치하고 로그인하세요. Codex 아이콘 또는 명령 팔레트의 `Codex: Open Codex Sidebar`로 엽니다. 공식 안내: https://learn.chatgpt.com/docs/ide

예시 요청: "이 저장소의 ImageCNN.forward에서 tensor shape가 어떻게 변하는지 설명해줘. 코드는 아직 바꾸지 마." / "ailab.doctor 실행 결과를 보고 GPU 설정 문제를 진단해줘."
