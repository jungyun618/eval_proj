"""노트북 코드 셀을 직접 실행해서 파이프라인을 검증하는 로컬 테스트 (GPU·API 키 불필요)

검증 내용
  1. 01 후처리: 정지 문자열 자르기, 코드블록 추출, 종료 사유(stop > eos > length)
  2. 02 채점기: 정답 코드 164개 전부 통과, 타임아웃·메모리 제한, 환경변수 격리
  3. 02 pass@1: 정답을 아는 합성 생성 결과로 판정 불일치 0건
  4. 02 judge: 가짜 judge로 smoke → sample → full → 재실행(완료분 건너뛰기) 흐름, 파싱 규칙
  5. 03 분석: 표 T1~T8 생성, 심어 둔 차이(정확한 judge vs 노이즈 judge, halo 유무)를 구분하는지

실행
  pip install numpy pandas scipy scikit-learn datasets
  python tests/test_pipeline.py                         # HumanEval을 Hugging Face에서 받음
  python tests/test_pipeline.py --humaneval HumanEval.jsonl.gz   # 오프라인: openai/human-eval 저장소의 data 파일

검증하지 않는 것: 01의 GPU 생성 셀, 02의 실제 API 호출 (Colab smoke 실행으로 확인)
"""
import argparse, glob, gzip, json, os, random, sys, tempfile, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
CRIT = ["correctness", "readability", "efficiency", "robustness"]


def cell(notebook, marker):
    """첫 줄이 marker로 시작하는 코드 셀의 소스"""
    nb = json.load(open(os.path.join(NB, notebook)))
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith(marker)]
    assert len(hits) == 1, f"{notebook}: '{marker}' 셀 {len(hits)}개"
    return "".join(hits[0]["source"])


def run(notebook, marker, ns):
    exec(compile(cell(notebook, marker), f"{notebook}:{marker}", "exec"), ns)


def load_problems(path):
    if path:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            return [json.loads(l) for l in f if l.strip()]
    from datasets import load_dataset
    return list(load_dataset("openai/openai_humaneval", split="test"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humaneval", help="HumanEval.jsonl(.gz) 경로. 없으면 Hugging Face에서 받음")
    ap.add_argument("--keep", action="store_true", help="임시 결과 폴더를 지우지 않고 남김(표 확인용)")
    args = ap.parse_args()

    if "tqdm" not in sys.modules:           # 노트북은 tqdm을 쓰지만 테스트에는 필요 없음
        try:
            import tqdm.auto  # noqa: F401
        except ImportError:
            stub = types.ModuleType("tqdm"); auto = types.ModuleType("tqdm.auto")
            auto.tqdm = lambda it=None, **kw: it
            stub.auto = auto
            sys.modules["tqdm"], sys.modules["tqdm.auto"] = stub, auto

    problems = load_problems(args.humaneval)
    P = {p["task_id"]: p for p in problems}
    assert len(P) == 164, len(P)
    run_dir = tempfile.mkdtemp(prefix="code_eval_test_")
    os.makedirs(f"{run_dir}/generations")
    print("임시 RUN_DIR:", run_dir)

    # ---------- 1. 01 후처리 ----------
    ns1 = {}
    run("01_generate.ipynb", "# 후처리", ns1)
    pp = ns1["postprocess"]
    p0 = P["HumanEval/0"]
    r = pp("raw", p0, "\n" + p0["canonical_solution"] + "\n\n\ndef extra():\n    pass\n", 80, False, 2048)
    assert r["finish_reason"] == "stop" and "extra" not in r["exec_code"] and r["syntax_ok"], r
    assert pp("raw", p0, "    return False\n", 5, True, 2048)["finish_reason"] == "eos"
    assert pp("raw", p0, "    x = 1\n" * 50, 2048, False, 2048)["finish_reason"] == "length"
    assert not pp("raw", p0, "", 1, True, 2048)["extraction_ok"]
    full = p0["prompt"] + p0["canonical_solution"]
    r = pp("chat", p0, "```python\n" + full + "```", 300, False, 2048)
    assert r["finish_reason"] == "stop" and r["extraction_ok"] and r["syntax_ok"] and "```" not in r["judge_code"]
    r = pp("chat", p0, "Here:\n```python\n" + full[:-30], 2048, False, 2048)     # 닫는 펜스 없이 한도 도달
    assert r["finish_reason"] == "length" and r["extraction_ok"] and not r["syntax_ok"]
    r = pp("chat", p0, full, 300, True, 2048)                                      # 펜스 없음
    assert not r["extraction_ok"] and r["finish_reason"] == "eos"
    assert not pp("chat", p0, "```python\ndef other(x):\n    return x\n```", 30, False, 2048)["extraction_ok"]
    print("1. 후처리 OK")

    # ---------- 합성 생성 결과 (정답 여부를 알고 있음) ----------
    rng = random.Random(1)
    truth = {}
    keys = {"base-a": ("raw", 0.6), "base-b": ("raw", 0.35), "chat-c": ("chat", 0.85)}
    for key, (mode, p_good) in keys.items():
        path = f"{run_dir}/generations/{key}.jsonl"
        with open(path, "w") as f:
            for p in problems:
                good = rng.random() < p_good
                body = p["canonical_solution"] if good else "    return None\n"
                raw = ("\n" + body + "\n\n\ndef helper():\n    pass\n") if mode == "raw" else ("```python\n" + p["prompt"] + body + "```")
                rec = {"task_id": p["task_id"], "model_key": key, "model_id": "x", "mode": mode,
                       "prompt_version": "v", "prompt_sent": "", "raw_response": raw,
                       **pp(mode, p, raw, 100, False, 2048), "n_new_tokens": 100,
                       "max_new_tokens": 2048, "gen_seconds": 1.0, "created_at": ""}
                f.write(json.dumps(rec) + "\n")
                truth[(key, p["task_id"])] = good
        with open(path) as f:                                  # 중복 레코드 1건 → 첫 레코드만 쓰는지 확인
            first = f.readline()
        with open(path, "a") as f:
            f.write(first)

    # ---------- 2·3. 02 채점기와 pass@1 ----------
    ns2 = {"os": os, "json": json, "glob": glob, "RUN_DIR": run_dir, "GEN_DIR": f"{run_dir}/generations", "PROBLEMS": P}
    run("02_evaluate.ipynb", "# 생성 결과 로드", ns2)
    assert all(len(v) == 164 for v in ns2["GENS"].values())
    run("02_evaluate.ipynb", "# 채점기 검증", ns2)
    rp = ns2["run_program"]
    assert rp("while True: pass", timeout=2)[1] == "Timeout"
    assert rp("x = bytearray(3 * 1024**3)")[1] == "MemoryError"
    os.environ["CODE_EVAL_TEST_SECRET"] = "leak"
    assert rp("import os, sys; sys.exit(1 if os.environ.get('CODE_EVAL_TEST_SECRET') else 0)")[0], "환경변수가 전달됨"
    print("2. 채점기 OK")
    run("02_evaluate.ipynb", "# pass@1", ns2)
    pres = [json.loads(l) for l in open(f"{run_dir}/pass_results.jsonl")]
    mism = [r for r in pres if r["passed"] != truth[(r["model_key"], r["task_id"])]]
    # 'return None'이 우연히 통과하는 문제는 허용 (정답인데 실패로 판정되는 경우만 오류)
    assert all(not truth[(r["model_key"], r["task_id"])] and r["passed"] for r in mism), mism[:3]
    print(f"3. pass@1 OK (우연 통과 {len(mism)}건)")

    # ---------- 4. 02 judge (가짜 provider) ----------
    calls = {"n": 0}

    def fake_call(cfg, prompt):
        calls["n"] += 1
        code = prompt.rsplit("[코드]", 1)[1]
        good = "return None" not in code
        kind = cfg["model_id"]
        rr = random.Random(hash((prompt, kind, calls["n"] % 7)))
        if kind == "bad-format" and rr.random() < 0.1:
            return {"text": "점수는 4점입니다", "model_returned": kind, "stop": "end", "truncated": False,
                    "usage": {"input": 900, "output": 50, "thinking": None}}
        if rr.random() < 0.02:
            raise RuntimeError("503 unavailable")                 # 재시도 경로 확인
        noise = {"good": 0.6, "noisy": 1.4, "bad-format": 0.9}[kind]
        base = 4.5 if good else 2.0
        sc = {c: int(min(5, max(1, round(base + rr.gauss(0, noise))))) for c in CRIT}
        sc["readability"] = int(min(5, max(1, round(4 + rr.gauss(0, 0.7)))))   # 정답과 무관한 항목(halo 없음)
        body = json.dumps({c: {"reason": "r", "score": v} for c, v in sc.items()}, ensure_ascii=False)
        return {"text": "```json\n" + body + "\n```", "model_returned": kind + "-2026", "stop": "end",
                "truncated": False, "usage": {"input": 900, "output": 300, "thinking": None}}

    fake = lambda mid, price: {"provider": "fake", "model_id": mid, "max_tokens": 1, "price": price, "kwargs": {}}
    ns2.update({"JUDGE_MODE": "smoke", "SAMPLE_TASKS": 5, "REPEAT_N": 20, "N_WORKERS": 8, "SEED": 0,
                "JUDGE_KEYS": None, "JUDGE_PATH": f"{run_dir}/judge_results.jsonl", "API_KEYS": {},
                "JUDGES": {"judge-good": fake("good", (2, 10)), "judge-noisy": fake("noisy", (1, 5)),
                           "judge-fmt": fake("bad-format", (0.5, 2))}})
    run("02_evaluate.ipynb", "# judge 함수", ns2)
    ns2["PROVIDERS"]["fake"] = fake_call
    ns2["time"].sleep = lambda s: None

    pj = ns2["parse_judge"]
    assert pj('{"correctness":{"score":5},"readability":{"score":"4"},"efficiency":{"score":3.0},"robustness":{"score":1}}')["readability"] == 4
    for bad in ['{"correctness":{"score":6},"readability":{"score":4},"efficiency":{"score":3},"robustness":{"score":1}}',
                'no json',
                '{"correctness":{"score":4.5},"readability":{"score":4},"efficiency":{"score":3},"robustness":{"score":1}}']:
        try:
            pj(bad)
        except Exception:
            continue
        raise AssertionError("파싱이 실패해야 함: " + bad)
    assert "{x}" in ns2["build_prompt"]("def f():", "d = {x}")

    for mode in ["smoke", "sample", "full", "full"]:     # 마지막 full: 남은 작업 0건이어야 함
        ns2["JUDGE_MODE"] = mode
        run("02_evaluate.ipynb", "# 실행", ns2)
    lines = [json.loads(l) for l in open(f"{run_dir}/judge_results.jsonl")]
    assert len(lines) == 164 * 3 * 3 + 20 * 3, len(lines)
    print("4. judge OK")

    # ---------- 5. 03 분석 ----------
    ns3 = {"os": os, "json": json, "glob": glob, "RUN_DIR": run_dir}
    for m in ["# 로드 + 공통 함수", "# T1", "# T3", "# T5", "# T7"]:
        run("03_analyze.ipynb", m, ns3)
    import pandas as pd
    a = f"{run_dir}/analysis"
    T4 = pd.read_csv(f"{a}/T4_judge_validity.csv", index_col=0)
    assert T4.loc["judge-good", "AUC"] > 0.9 and T4.loc["judge-good", "AUC"] > T4.loc["judge-noisy", "AUC"] + 0.05
    T7 = pd.read_csv(f"{a}/T7_halo.csv", index_col=0)
    assert abs(T7.loc["judge-good", "readability~correctness_rho"]) < 0.2      # halo 없게 심은 항목
    assert T7.loc["judge-good", "efficiency~correctness_rho"] > 0.5           # halo 있게 심은 항목
    assert len(glob.glob(f"{a}/*.csv")) >= 11
    print("5. 분석 OK")
    if not args.keep:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
    print("\n전체 통과")


if __name__ == "__main__":
    main()
