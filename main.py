'''
태스크 로드: tasks.json → LLM 호출 프롬프트로 변환

결과/모델 폴더: 자동 생성

학습 실행: Random / Q-learning / Q-RL / DQN

시각화: 평균 보상 그래프 저장 (results/episode_rewards.png)

평가 지표 출력 & CSV 저장: 평균 보상, 총 전력, 총 탄소, 평균 처리 시간, Baseline 대비 개선율

모델 저장: DQN, Q-RL, Q-learning 학습 모델 저장 (models/ 폴더)
'''

import json
import os
import csv
import matplotlib.pyplot as plt
import torch
from env import LLMTaskEnv
from train import run_experiment

API_KEY = "YOUR_OPENAI_API_KEY"

# === 1. 태스크 로드 ===
env = LLMTaskEnv(tasks_path="data/tasks.json", out_dir="results")
episodes = len(env.tasks)

# === 2. 결과/모델 폴더 생성 ===
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# === 3. 각 방법 실행 ===
random_results = run_experiment(env, episodes, "random")
rl_results = run_experiment(env, episodes, "qlearning")
qrl_results = run_experiment(env, episodes, "qrl")
dqn_results = run_experiment(env, episodes, "dqn")

agents_results = {
    "Random": random_results,
    "Q-learning": rl_results,
    "Q-RL": qrl_results,
    "DQN": dqn_results
}

# === 4. 평균 보상 시각화 및 저장 ===
results_rewards = {name: res["total_rewards"] for name, res in agents_results.items()}

for name, rewards in results_rewards.items():
    plt.plot(rewards, label=name)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards by Agent")
plt.legend()
plt.savefig("results/episode_rewards.png")
plt.close()

# === 5. 평가 지표 계산 및 출력 ===
baseline_power = random_results["total_power"][-1]
baseline_carbon = random_results["total_carbon"][-1]

with open("results/evaluation.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Agent", "Avg Reward", "Total Power (kWh)", "Power Improvement (%)",
        "Total Carbon (kgCO2)", "Carbon Improvement (%)", "Avg Task Time (sec)"
    ])
    
    for name, res in agents_results.items():
        total_power = res["total_power"][-1]
        total_carbon = res["total_carbon"][-1]
        avg_time = res["avg_time"][-1]
        avg_reward = sum(res["total_rewards"]) / len(res["total_rewards"])
        power_improvement = (baseline_power - total_power) / baseline_power * 100
        carbon_improvement = (baseline_carbon - total_carbon) / baseline_carbon * 100
        
        # 화면 출력
        print(f"[{name}]")
        print(f"  평균 보상: {avg_reward:.4f}")
        print(f"  총 전력 소모: {total_power:.4f} kWh (개선율: {power_improvement:.2f}%)")
        print(f"  총 탄소 배출량: {total_carbon:.4f} kgCO2 (개선율: {carbon_improvement:.2f}%)")
        print(f"  평균 처리 시간: {avg_time:.4f} sec\n")
        
        # CSV 저장
        writer.writerow([
            name, avg_reward, total_power, power_improvement,
            total_carbon, carbon_improvement, avg_time
        ])

# === 6. 학습된 모델 저장 ===
# DQN 예시
if "dqn" in locals() and hasattr(dqn_results, "agent"):
    torch.save(dqn_results["agent"].policy_net.state_dict(), "models/dqn_final.pth")

# Q-RL, Q-learning Q-table 저장 예시
if "qrl" in locals() and hasattr(qrl_results, "agent"):
    torch.save(qrl_results["agent"].q_table, "models/qrl_final.pth")

if "rl" in locals() and hasattr(rl_results, "agent"):
    torch.save(rl_results["agent"].q_table, "models/qlearning_final.pth")

print("모든 학습 완료 및 결과/모델 저장 완료.")