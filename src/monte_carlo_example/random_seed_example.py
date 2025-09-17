import numpy as np


def random_seed_example(
    main_seed: int, number_of_simulations: int, years_to_simulate: int
) -> None:

    # Localized rng with own seed
    main_rng = np.random.default_rng(seed=main_seed)

    # Limit to integers between 1 and 100 for simplicity
    principle_seeds = main_rng.integers(
        low=1, high=100, size=number_of_simulations
    ).tolist()

    print(f"Main Seed: {main_seed}")
    print(f"Number of Simulations: {number_of_simulations}")
    print(f"Years to Simulate: {years_to_simulate}")
    print("-" * 50)
    print(f"Principle Seeds (seed={main_seed}): {principle_seeds}")
    print("-" * 50)

    year = 2025

    for idx, current_seed in enumerate(principle_seeds):
        child_rng = np.random.default_rng(seed=current_seed)

        yearly_seeds = child_rng.integers(
            low=1, high=100, size=years_to_simulate
        ).tolist()

        print(f"Simulation #{idx + 1} (seed={current_seed}): {yearly_seeds}")


if __name__ == "__main__":
    # Change the arguments to see different example print outs
    random_seed_example(main_seed=42, number_of_simulations=20, years_to_simulate=5)
