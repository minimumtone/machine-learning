"""Tests for t2vasp.optimizer — GA and evaluation functions."""

import pytest

from t2vasp.optimizer import (
    Candidate,
    GAConfig,
    GeneticAlgorithm,
    energy_fitness,
    weighted_fitness,
)


class TestEnergyFitness:
    def test_nested_dict(self) -> None:
        d = {"energy": {"energy_per_atom": -5.0}}
        assert energy_fitness(d) == -5.0

    def test_missing_returns_inf(self) -> None:
        assert energy_fitness({}) == float("inf")


class TestWeightedFitness:
    def test_single_weight(self) -> None:
        d = {"x": 3.0}
        assert weighted_fitness(d, weights={"x": 2.0}) == 6.0

    def test_multi_weight(self) -> None:
        d = {"x": 3.0, "y": -1.0}
        assert weighted_fitness(d, weights={"x": 1.0, "y": 1.0}) == 2.0


class TestGeneticAlgorithm:
    def test_finds_minimum_of_quadratic(self) -> None:
        """GA should find the minimum of f(x) = (x-3)^2 near x=3."""

        def eval_fn(params):
            return (params["x"] - 3.0) ** 2

        config = GAConfig(
            population_size=30,
            num_generations=40,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elite_fraction=0.1,
            seed=42,
            param_bounds={"x": (-10.0, 10.0)},
        )
        ga = GeneticAlgorithm(config, eval_fn)
        best = ga.run()
        assert abs(best.params["x"] - 3.0) < 0.5

    def test_top_candidates(self) -> None:
        def eval_fn(params):
            return params["a"] ** 2

        config = GAConfig(
            population_size=10,
            num_generations=5,
            seed=1,
            param_bounds={"a": (-5.0, 5.0)},
        )
        ga = GeneticAlgorithm(config, eval_fn)
        ga.run()
        top = ga.top_candidates(3)
        assert len(top) == 3
        assert top[0].fitness <= top[1].fitness <= top[2].fitness
