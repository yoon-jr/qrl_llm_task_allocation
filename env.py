import gym
from gym import spaces
import numpy as np
import time
import openai

import argparse
import json as jsonlib
import time
from pathlib import Path
import re

import openai
import os

# --- 공통 유틸 ---
from _common_eval import is_correct
from _common_log import RunLogger

# ---------- 데이터 로드 ----------
def load_json_or_jsonl(path: Path):
    data = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(jsonlib.loads(line))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = jsonlib.load(f)
            if isinstance(payload, list):
                data = payload
            else:
                raise ValueError("JSON top-level must be a list")
    else:
        raise ValueError("Only .json or .jsonl supported")
    return data

# 단계별 풀이 및 JSON 정답 출력 지시 프롬프트
PROMPT_KO = '''당신은 수학 전문가입니다. 주어진 문제를 단계별로 해결하세요.
- 조건과 알려진 값을 식으로 정리하세요.
- 필요한 공식을 적용해 풀이하세요.
- 단계별로 계산하고 중간 결과를 적으세요.
- 마지막에 정답이 맞는지 다시 검산하세요.
- 출력은 반드시 JSON 형식만 작성해야 합니다. 설명, 불필요한 텍스트, "출력:" 같은 접두어는 절대 포함하지 마세요.
- JSON은 반드시 { 로 시작하고 } 로 끝나야 하며, 키는 아래와 같이 두 개만 사용하세요:
{
  "StepByStepSolution": "1) ... 2) ... 3) ...",
  "FinalAnswer": <정답>
}
- FinalAnswer는 반드시 정수만 출력하세요. 단위(예: km, 개), null, None, 빈 문자열, 불가능 같은 값은 절대 쓰지 마세요.
- 출력은 반드시 한국어로 작성하세요.
- FinalAnswer이 나왔다면 생성을 중단하고 출력하세요.
- 풀이가 틀렸다고 생각되면 이전 단계로 돌아가 다시 생각하여 정확하게 계산하고 반드시 정수 FinalAnswer를 출력하세요.
- 풀이 과정은 논리적으로 생각하되, 풀이 과정 생성은 최대한 간결하게 핵심만 작성하세요.
- 근거 없는 추정은 하지 말고 반드시 주어진 정보를 토대로 풀이하세요.
- 새로운 문제를 만들어내지 말고, 주어진 문제만 풀어야 합니다.
'''

# ---------- 정수 추출 ----------
def extract_pred_raw(text_out: str) -> str:
    cand = text_out.strip()
    if cand.startswith("```"):
        cand = re.sub(r"^```(?:json)?", "", cand).strip()
        if cand.endswith("```"):
            cand = cand[:-3].strip()
    try:
        parsed = jsonlib.loads(cand)
        for k in ["FinalAnswer","final_answer","answer","final","Final"]:
            if k in parsed:
                return str(parsed[k])
    except Exception:
        pass
    # fallback: 마지막 줄 중 숫자 포함
    lines = [ln.strip() for ln in cand.split("\n") if ln.strip()]
    for ln in reversed(lines):
        match = re.search(r"-?\d+", ln)
        if match:
            return match.group()
    return ""

class LLMTaskEnv(gym.Env):
    def __init__(self, tasks_path=None, out_dir="results", carbon_intensity=0.47, power_per_token=0.00005, max_tokens=1024, temperature=0.0, top_p=1.0, top_k=0):
        super().__init__()

        self.tasks = []
        if tasks_path:
            tasks_data = load_json_or_jsonl(Path(tasks_path))
            # 문제만 뽑아서 tasks 리스트에 저장
            self.tasks = [item["problem"] for item in tasks_data]
        self.num_tasks = len(self.tasks)
        self.remaining_tasks = self.num_tasks
        
        # 평가 지표 계산을 위한 계수
        self.carbon_intensity = carbon_intensity  # kgCO2/kWh
        self.power_per_token = power_per_token    # 토큰당 전력 소비량 (kWh)

        # --- LLM 설정 (Gemini) ---
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = "gemini-2.5-flash"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # --- 로그 ---
        self.logger = RunLogger(Path(out_dir))

        # --- 누적값 초기화 ---
        self.total_power = 0.0
        self.total_carbon = 0.0
        self.total_time = 0.0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.total_samples = 0
        self.n_correct = 0
        self.peak_mem_bytes_max = 0
        
        # 관측/행동 공간
        self.observation_space = spaces.Box(low=0, high=self.num_tasks, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)  # 단일 LLM 호출

    def reset(self):
        self.remaining_tasks = self.num_tasks
        self.last_time = 0.0
        self.last_tokens = 0
        self.total_power = 0.0
        self.total_carbon = 0.0
        self.total_time = 0.0
        return np.array([self.remaining_tasks, self.last_time, self.last_tokens], dtype=np.float32)
    
    def step(self, action=None):
        task_idx = self.num_tasks - self.remaining_tasks
        problem = self.tasks[task_idx]
        gold = None  # 만약 gold answer가 필요하면 tasks에 포함

        prompt_text = f"{PROMPT_KO}\n\n문제:\n{problem}"
        parts = [prompt_text]

        def _extract_text(resp) -> str:
            texts = []
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                if hasattr(content, "parts"):
                    for p in content.parts:
                        if hasattr(p, "text") and p.text:
                            texts.append(p.text)
            return "\n".join(texts).strip()

        # LLM 호출
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text_out = _extract_text(resp)
        except Exception as e:
            print(f"⚠️ Gemini 호출 오류: {e}")
            text_out = ""

        elapsed = time.perf_counter() - start
        tokens_used = len(text_out.split())
        pred_raw = extract_pred_raw(text_out)
        correct = is_correct(pred_raw, gold) if gold else None

        # 로그 기록
        self.logger.log_generation(ex_id=task_idx, problem=problem,
                                   gold_answer=gold, model_output=text_out)
        self.logger.log_sample_metrics(ex_id=task_idx, latency_sec=elapsed,
                                       output_tokens=tokens_used, correct=correct)
        
        # 전력/탄소/보상 계산
        power = tokens_used * self.power_per_token
        carbon = power * self.carbon_intensity
        reward = (1.0 if correct else 0.0) - 0.01 * elapsed - carbon # 정확도, 지연, 탄소 배출 동시 반영한 보상

        self.total_power += power
        self.total_carbon += carbon
        self.total_time += elapsed
        self.remaining_tasks -= 1

        obs = [self.remaining_tasks, elapsed, tokens_used]
        done = self.remaining_tasks == 0
        info = {"output": text_out, "power": power, "carbon": carbon, "time": elapsed, "correct": correct}

        return obs, reward, done, info