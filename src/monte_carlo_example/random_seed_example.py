import numpy as np


def random_seed_example(
    main_seed: int, number_of_simulations: int, years_to_simulate: int
) -> None:
    """
    Demonstrates hierarchical seeding using numpy random number generator.

    This function simulates a multi-level random number generation process.
    A main random seed is used to generate a set of "principle seeds" for each simulation.
    Each simulation then uses its own seed to generate yearly seeds for a defined number of years.

    Parameters:
        main_seed (int): The top-level seed used to initialize the main RNG.
        number_of_simulations (int): Number of independent simulations to generate.
        years_to_simulate (int): Number of years in each simulation.

    Returns:
        None: Prints the generated seeds to the console.

    Examples:
        >>> random_seed_example(main_seed=42, number_of_simulations=2, years_to_simulate=5)
        Main Seed: 42
        Number of Simulations: 2
        Years to Simulate: 5
        --------------------------------------------------
        Principle Seeds (seed=42): [9, 77]
        --------------------------------------------------
        Simulation #1 (seed=9): [42, 87, 96, 29, 12]
        Simulation #2 (seed=77): [6, 78, 63, 55, 79]
    """

    # Localized rng with own seed
    main_rng = np.random.default_rng(seed=main_seed)

    # Get a list of principal seeds for the number of desired simulations
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

    # For each primary simulation seed, generate a list of seeds for the desired number of..
    # ..years to be simulated for each simulation
    for idx, current_seed in enumerate(principle_seeds):
        child_rng = np.random.default_rng(seed=current_seed)

        # Technically not accounting for (n + 1) for initial year here, eh
        yearly_seeds = child_rng.integers(
            low=1, high=100, size=years_to_simulate
        ).tolist()

        print(f"Simulation #{idx + 1} (seed={current_seed}): {yearly_seeds}")


if __name__ == "__main__":
    # Change the arguments to see different example print outs
    random_seed_example(main_seed=42, number_of_simulations=2, years_to_simulate=5)
