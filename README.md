# code-eval

HumanEval에서 LLM이 생성한 코드를 **실행 채점(pass@1)**과 **LLM-as-judge**로 함께 평가해, judge가 실제 정답 여부와 얼마나 맞는지, 그리고 judge끼리 얼마나 일치하는지를 측정한다.

## 연구 질문
1. judge의 correctness 점수는 실제 테스트 통과 여부와 맞는가 (AUC, Spearman)
2. 서로 다른 judge는 같은 코드에 비슷한 점수를 주는가 (항목별 weighted kappa)
3. 정답이 없는 항목(가독성·효율성·견고성)이 correctness에 끌려가는가 (halo)

## 구성
| 노트북 | 런타임 | 하는 일 |
|---|---|---|
| `notebooks/01_generate.ipynb` | Colab GPU (T4) | 로컬 모델로 코드 생성. 정지·코드 추출·종료 사유 판정을 여기서 한 번만 수행 |
| `notebooks/02_evaluate.ipynb` | Colab CPU | 정답 코드 164개로 채점기 검증 → pass@1 → LLM judge 채점 |
| `notebooks/03_analyze.ipynb` | Colab CPU | 두 결과를 처음으로 합쳐 표 T1~T8 생성 |

pass@1과 judge는 서로의 결과를 보지 않으며, 03에서만 `(model_key, task_id)`로 합쳐진다. judge에는 생성 모델 이름을 주지 않는다.

## 설정
- **데이터**: HumanEval 164문제 (`openai/openai_humaneval`)
- **생성 모델**: StarCoder2-3B, CodeLlama-7B(프롬프트 이어쓰기), Qwen2.5-7B-Instruct(채팅 템플릿 + 함수 전체 생성)
  - 공통: 8-bit 양자화, greedy, `max_new_tokens` 2048
- **judge**: Claude, GPT, Gemini 계열 API 모델
  - 정확성·가독성·효율성·견고성 각 1~5점, JSON 출력(rubric r2)
- **pass@1 실행 환경**: 별도 프로세스(`python -I`), 최소 환경변수, 메모리 2GB, 10초 타임아웃. API 키를 읽기 전에 실행한다.

## 실행
API 키는 Colab Secrets에 등록한다: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`(선택: `HF_TOKEN`).

1. 01을 `RUN_NAME="smoke"`(5문제)로 실행
2. 02 `smoke`로 judge 모델 ID와 응답 형식 확인
3. 01을 `full`로 실행
4. 02 `sample`로 비용 추정
5. 02 `full`
6. 03

결과는 Drive `MyDrive/code_eval_v4/<RUN_NAME>/`에 jsonl로 저장된다. 문제 1개 단위로 저장되므로 세션이 끊겨도 다시 실행하면 이어서 진행된다.

## 로컬 테스트
노트북의 코드 셀을 그대로 실행해 후처리, 채점기, pass@1 판정, judge 흐름(가짜 judge), 분석 표를 검증한다. GPU와 API 키는 필요 없다.

```bash
pip install numpy pandas scipy scikit-learn datasets
python tests/test_pipeline.py
# 오프라인: python tests/test_pipeline.py --humaneval path/to/HumanEval.jsonl.gz
```

GPU 생성 셀과 실제 API 호출은 이 테스트로 검증되지 않는다. Colab smoke 실행으로 확인한다.

## 알려진 한계
- **생성 프로토콜이 다름**: Qwen(지시문 + 함수 전체 생성 + 코드블록 추출)과 베이스 모델(본문 이어쓰기)의 프로토콜이 달라서, 모델 간 pass@1 순위는 참고용이다.
- **보조 함수 절단**: 이어쓰기 모드의 정지 문자열(`\ndef ` 등, Codex 관례)은 본 함수 뒤에 정의된 최상위 보조 함수를 잘라낸다.
- **정답 없는 항목**: 가독성·효율성·견고성은 정답 기준이 없어 judge 간 일치도만 보고한다.
- **데이터 오염**: HumanEval은 학습 데이터 오염 가능성이 있으므로, 절대 점수보다 조건 간 상대 비교로 해석한다.
