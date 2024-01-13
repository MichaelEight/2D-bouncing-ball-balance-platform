import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import re

def load_average_rewards(model_dir, iteration):
    avg_rewards_path = os.path.join(model_dir, f'average_rewards_iteration_{iteration}.npy')
    if os.path.exists(avg_rewards_path):
        return np.load(avg_rewards_path)
    else:
        print("Average rewards file not found.")
        return None

def find_latest_rewards_file(model_dir):
    reward_files = [f for f in os.listdir(model_dir) if 'average_rewards' in f and f.endswith('.npy')]
    if not reward_files:
        return None

    def extract_iteration(filename):
        match = re.search(r'average_rewards_iteration_(\d+).npy', filename)
        return int(match.group(1)) if match else -1

    latest_file = max(reward_files, key=extract_iteration)
    return os.path.join(model_dir, latest_file)

def update_graph(frame, model_dir, line, ax, fig):
    rewards_file_path = find_latest_rewards_file(model_dir)
    if rewards_file_path:
        average_rewards = np.load(rewards_file_path)
        if average_rewards is not None:
            line.set_ydata(average_rewards)
            line.set_xdata(range(len(average_rewards)))
            ax.relim()  # Recalculate limits
            ax.autoscale_view(True,True,True)  # Rescale the view based on the new data limits

            # Update the axes title with the latest iteration number
            latest_iteration = len(average_rewards) - 1
            ax.set_title(f'Average Reward Rates Over Iterations - Iteration: {latest_iteration}')
            fig.canvas.draw_idle()
    return line,


def main():
    model_dir = 'saved_modelsv4/'
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_title('Average Reward Rates Over Iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')

    ani = FuncAnimation(fig, update_graph, fargs=(model_dir, line, ax, fig),
                        interval=30000, blit=False)

    plt.show()

if __name__ == "__main__":
    main()