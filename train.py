import numpy as np
from agents.qlearning import QLearningAgent
from agents.qrl import QRLAgent
from agents.dqn import DQNAgent

def run_experiment(env, episodes, agent_type="random"):
    # 결과 저장용
    results = {
        "total_rewards": [],
        "total_power": [],
        "total_carbon": [],
        "avg_time": []
    }

    if agent_type == "qlearning":
        agent = QLearningAgent()
    elif agent_type == "qrl":
        agent = QRLAgent()
    elif agent_type == "dqn":
        agent = DQNAgent(state_dim=3, action_dim=1)
    else:
        agent = None
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0.0
        while not done:
            if agent_type == "random":
                action = 0
            else:
                action = agent.choose_action(state)
            
            next_state, reward, done, info = env.step(action)

            if agent_type in ["qlearning", "qrl"]:
                agent.learn(state, action, reward, next_state)
            elif agent_type == "dqn":
                agent.buffer.push(state, action, reward, next_state, done)
                agent.learn(batch_size=2)
            
            total_r += reward
            state = next_state

        # 에피소드 결과 저장
        results["total_rewards"].append(total_r)
        results["total_power"].append(env.total_power)
        results["total_carbon"].append(env.total_carbon)
        results["avg_time"].append(env.total_time / env.num_tasks)

    return results