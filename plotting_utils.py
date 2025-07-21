import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


class PlotManager:
    def __init__(self, output_folder="plots"):
        """
        Initialize PlotManager with output folder for saving plots

        Args:
            output_folder (str): Folder to save all plots
        """
        self.output_folder = output_folder
        self.ensure_folder_exists()

    def ensure_folder_exists(self):
        """Create output folder if it doesn't exist"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created plots folder: {self.output_folder}")

    def save_plot(self, filename, dpi=300, bbox_inches="tight"):
        """
        Save current plot to file

        Args:
            filename (str): Name of the file to save
            dpi (int): DPI for the saved image
            bbox_inches (str): Bounding box setting
        """
        filepath = os.path.join(self.output_folder, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved plot: {filepath}")

    def plot_reward_curve(
        self, rewards, algorithm_name="DQN", game_name="Galaxian", save_plot=True
    ):
        """
        Plot reward curve over episodes

        Args:
            rewards (list): List of rewards per episode
            algorithm_name (str): Name of the algorithm (DQN/DDQN)
            game_name (str): Name of the game
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, linewidth=1.5, alpha=0.8)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(
            f"{algorithm_name} Training Progress on {game_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)

        # Add moving average
        if len(rewards) > 10:
            window = min(10, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(rewards)),
                moving_avg,
                linewidth=2,
                color="red",
                alpha=0.8,
                label=f"Moving Average (window={window})",
            )
            plt.legend()

        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{algorithm_name.lower()}_{game_name.lower()}_rewards_{timestamp}.png"
            )
            self.save_plot(filename)

        plt.show()

    def plot_q_value_estimates(
        self,
        avg_max_q,
        eval_points,
        algorithm_name="DQN",
        game_name="Galaxian",
        save_plot=True,
    ):
        """
        Plot average max Q-value estimates over time

        Args:
            avg_max_q (list): List of average max Q-values
            eval_points (list): Episode numbers where Q-values were evaluated
            algorithm_name (str): Name of the algorithm (DQN/DDQN)
            game_name (str): Name of the game
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            eval_points[: len(avg_max_q)],
            avg_max_q,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Max Q-Value", fontsize=12)
        plt.title(
            f"{algorithm_name} Q-Value Estimates Over Training on {game_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{algorithm_name.lower()}_{game_name.lower()}_qvalues_{timestamp}.png"
            )
            self.save_plot(filename)

        plt.show()

    def plot_overestimation_bias(self, algorithm_name="DQN", save_plot=True):
        """
        Plot overestimation bias demonstration

        Args:
            algorithm_name (str): Name of the algorithm (DQN/DDQN)
            save_plot (bool): Whether to save the plot
        """
        num_actions_list = [2, 5, 10, 20, 50]
        num_samples = 10000

        if algorithm_name.upper() == "DQN":
            # DQN-like bias (higher overestimation)
            bias_values = []
            for num_actions in num_actions_list:
                true_q = np.zeros(num_actions)
                errors = np.random.normal(0, 1, (num_samples, num_actions))
                max_q = np.max(true_q + errors, axis=1)
                bias_values.append(np.mean(max_q))
            label = "DQN-like Bias"
            color = "red"
            marker = "o"
        else:
            # DDQN-like bias (lower overestimation)
            bias_values = []
            for num_actions in num_actions_list:
                true_q = np.zeros(num_actions)
                errors_a = np.random.normal(0, 1, (num_samples, num_actions))
                errors_b = np.random.normal(0, 1, (num_samples, num_actions))
                argmax_a = np.argmax(true_q + errors_a, axis=1)
                dq_values = (true_q + errors_b)[np.arange(num_samples), argmax_a]
                bias_values.append(np.mean(dq_values))
            label = "DDQN-like Bias"
            color = "blue"
            marker = "x"

        plt.figure(figsize=(12, 6))
        plt.plot(
            num_actions_list,
            bias_values,
            label=label,
            marker=marker,
            linewidth=2,
            markersize=8,
            color=color,
        )
        plt.xlabel("Number of Actions", fontsize=12)
        plt.ylabel("Average Bias (Overestimation)", fontsize=12)
        plt.title(
            f"Overestimation Bias in Toy Environment ({algorithm_name}-like)",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name.lower()}_overestimation_bias_{timestamp}.png"
            self.save_plot(filename)

        plt.show()

    def plot_comparison(
        self, dqn_rewards, ddqn_rewards, game_name="Galaxian", save_plot=True
    ):
        """
        Plot comparison between DQN and DDQN performance

        Args:
            dqn_rewards (list): DQN rewards
            ddqn_rewards (list): DDQN rewards
            game_name (str): Name of the game
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(14, 7))

        # Plot individual reward curves
        plt.subplot(1, 2, 1)
        plt.plot(dqn_rewards, label="DQN", alpha=0.7, linewidth=1.5)
        plt.plot(ddqn_rewards, label="DDQN", alpha=0.7, linewidth=1.5)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Reward Comparison on {game_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot moving averages for better comparison
        plt.subplot(1, 2, 2)
        window = min(10, len(dqn_rewards) // 10)

        if len(dqn_rewards) >= window:
            dqn_avg = np.convolve(dqn_rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(dqn_rewards)),
                dqn_avg,
                label=f"DQN (avg)",
                linewidth=2,
            )

        if len(ddqn_rewards) >= window:
            ddqn_avg = np.convolve(ddqn_rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(ddqn_rewards)),
                ddqn_avg,
                label=f"DDQN (avg)",
                linewidth=2,
            )

        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.title(f"Moving Average Comparison on {game_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dqn_vs_ddqn_comparison_{game_name.lower()}_{timestamp}.png"
            self.save_plot(filename)

        plt.show()

    def create_training_summary(
        self,
        rewards,
        avg_max_q,
        eval_points,
        algorithm_name="DQN",
        game_name="Galaxian",
        save_plot=True,
    ):
        """
        Create a comprehensive training summary with multiple plots

        Args:
            rewards (list): List of rewards per episode
            avg_max_q (list): List of average max Q-values
            eval_points (list): Episode numbers where Q-values were evaluated
            algorithm_name (str): Name of the algorithm (DQN/DDQN)
            game_name (str): Name of the game
            save_plot (bool): Whether to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Reward curve
        ax1.plot(rewards, linewidth=1.5, alpha=0.8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title(f"{algorithm_name} Reward Curve")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Q-value estimates
        ax2.plot(
            eval_points[: len(avg_max_q)],
            avg_max_q,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Max Q-Value")
        ax2.set_title(f"{algorithm_name} Q-Value Estimates")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Reward distribution
        ax3.hist(rewards, bins=20, alpha=0.7, edgecolor="black")
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"{algorithm_name} Reward Distribution")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Training statistics
        stats_text = f"""
        Training Statistics:
        - Total Episodes: {len(rewards)}
        - Average Reward: {np.mean(rewards):.2f}
        - Max Reward: {np.max(rewards):.2f}
        - Min Reward: {np.min(rewards):.2f}
        - Std Reward: {np.std(rewards):.2f}
        - Final Epsilon: {eval_points[-1] if eval_points else "N/A"}
        """
        ax4.text(
            0.1,
            0.5,
            stats_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        ax4.set_title(f"{algorithm_name} Training Summary")
        ax4.axis("off")

        plt.suptitle(
            f"{algorithm_name} Training Summary on {game_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{algorithm_name.lower()}_{game_name.lower()}_summary_{timestamp}.png"
            )
            self.save_plot(filename)

        plt.show()
