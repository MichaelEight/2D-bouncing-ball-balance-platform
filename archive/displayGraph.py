import numpy as np
import matplotlib.pyplot as plt
import os
import re

def load_average_rewards(model_dir, iteration):
    """
    Load the average rewards from the saved numpy file.
    """
    avg_rewards_path = os.path.join(model_dir, f'average_rewards_iteration_{iteration}.npy')
    if os.path.exists(avg_rewards_path):
        return np.load(avg_rewards_path)
    else:
        print("Average rewards file not found.")
        return None

def plot_average_rewards(average_rewards):
    """
    Plot the average rewards using matplotlib.
    """
    plt.plot(average_rewards)
    plt.title('Average Reward Rates Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.show()



def find_latest_rewards_file(model_dir):
    reward_files = [f for f in os.listdir(model_dir) if 'average_rewards' in f and f.endswith('.npy')]
    if not reward_files:
        return None

    # Extracting iteration number using regular expression
    def extract_iteration(filename):
        # The regex now looks for one or more digits following the underscore and preceding '.npy'
        match = re.search(r'average_rewards_iteration_(\d+).npy', filename)
        return int(match.group(1)) if match else -1

    # Find the file with the maximum iteration number
    latest_file = max(reward_files, key=extract_iteration)
    return os.path.join(model_dir, latest_file)




def main():
    model_dir = 'saved_modelsv3/'
    rewards_file_path = find_latest_rewards_file(model_dir)
    # force path
    #rewards_file_path = 'saved_modelsv3/average_rewards_iteration_200.npy'
    if rewards_file_path:
        average_rewards = np.load(rewards_file_path)
        # print(average_rewards)
        if average_rewards is not None:
            plot_average_rewards(average_rewards)
    else:
        print("No average rewards files found in the directory.")

if __name__ == "__main__":
    main()
