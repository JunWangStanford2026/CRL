import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(args):
    horizons = []
    vanilla_means, vanilla_stds = [], []
    grpo_means, grpo_stds = [], []
    hra_means, hra_stds = [], []
    compositional_means, compositional_stds = [], []

    horizon = args.min_horizon
    while horizon <= args.max_horizon:
        horizons.append(horizon)
        print(f"Retrieving data for horizon {horizon} ...")
        vanilla_data = np.load(f"minilang/results/{horizon}_compute_units/vanilla_rewards.npy")
        vanilla_rewards = np.mean(vanilla_data, axis=1)
        vanilla_means.append(np.mean(vanilla_rewards))
        vanilla_stds.append(np.std(vanilla_rewards))

        grpo_data = np.load(f"minilang/results/{horizon}_compute_units/grpo_rewards.npy")
        grpo_rewards = np.mean(grpo_data, axis=1)
        grpo_means.append(np.mean(grpo_rewards))
        grpo_stds.append(np.std(grpo_rewards))

        hra_data = np.load(f"minilang/results/{horizon}_compute_units/hra_rewards.npy")
        hra_rewards = np.mean(hra_data, axis=1)
        hra_means.append(np.mean(hra_rewards))
        hra_stds.append(np.std(hra_rewards))

        compositional_data = np.load(f"minilang/results/{horizon}_compute_units/compositional_rewards.npy")
        compositional_rewards = np.mean(compositional_data, axis=1)
        compositional_means.append(np.mean(compositional_rewards))
        compositional_stds.append(np.std(compositional_rewards))

        horizon *= 2

    plt.title("Minilang Model Performances Comparison")

    plt.errorbar(
        horizons,
        vanilla_means,
        yerr=vanilla_stds,
        fmt='-o',
        capsize=4,
        label='Vanilla Reinforce'
    )

    plt.errorbar(
        horizons,
        grpo_means,
        yerr=grpo_stds,
        fmt='-o',
        capsize=4,
        label='GRPO Reinforce'
    )

    plt.errorbar(
        horizons,
        hra_means,
        yerr=hra_stds,
        fmt='-o',
        capsize=4,
        label='HRA Reinforce'
    )

    plt.errorbar(
        horizons,
        compositional_means,
        yerr=compositional_stds,
        fmt='-o',
        capsize=4,
        label='Compositional GRPO'
    )

    plt.xlabel("Training Compute")
    plt.ylabel("Average Model Performance (with STD)")
    plt.legend()

    plt.savefig(f"{args.plots_dir}/results.png")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--min-horizon", default=3500, type=int)
    parser.add_argument("--max-horizon", default=224000, type=int)
    parser.add_argument("--plots-dir", type=str, default="minilang/visualizations")
    args = parser.parse_args()
    main(args)
