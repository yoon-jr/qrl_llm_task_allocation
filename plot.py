import matplotlib.pyplot as plt

def plot_rewards(results):
    plt.figure(figsize=(10,6))
    for label, values in results.items():
        plt.plot(values, label=label, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (-Time - Tokens)")
    plt.title("LLM Task Execution Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
