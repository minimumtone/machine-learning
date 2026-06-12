"""
Optimisation module — evaluation functions and genetic algorithm for
structure search (Phase 6: USPEX-like capabilities).

Design principle: the optimiser is agnostic to the physical model.
It works with *candidates* (arbitrary dicts) and an *evaluation function*
that maps a candidate to a scalar fitness.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EvalFunc = Callable[[Dict[str, Any]], float]


# ── Candidate wrapper ────────────────────────────────────────────────
@dataclass
class Candidate:
    """A single candidate in the population."""
    params: Dict[str, float]
    fitness: float = float("inf")
    generation: int = 0
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "fitness": self.fitness,
                "generation": self.generation, **self.params}


# ── Evaluation functions ─────────────────────────────────────────────
def energy_fitness(result_dict: Dict[str, Any]) -> float:
    """Default fitness: lower energy_per_atom is better.

    Parameters
    ----------
    result_dict : dict
        Must contain ``energy.energy_per_atom`` (flattened or nested).

    Returns
    -------
    float
        Fitness value (lower is better).
    """
    e = result_dict.get("energy", {})
    if isinstance(e, dict):
        val = e.get("energy_per_atom")
    else:
        val = result_dict.get("energy.energy_per_atom")

    if val is None:
        return float("inf")
    return float(val)


def weighted_fitness(
    result_dict: Dict[str, Any],
    weights: Dict[str, float] | None = None,
) -> float:
    """Weighted multi-objective fitness.

    Parameters
    ----------
    result_dict : dict
        Flattened analysis result.
    weights : dict
        ``{key: weight}``; keys should match flattened result keys.
        Positive weight → minimise; negative → maximise.
    """
    if weights is None:
        return energy_fitness(result_dict)
    score = 0.0
    for key, w in weights.items():
        val = result_dict.get(key)
        if val is not None:
            score += w * float(val)
    return score


# ── Genetic Algorithm ────────────────────────────────────────────────
@dataclass
class GAConfig:
    """Parameters for the genetic algorithm."""
    population_size: int = 20
    num_generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_fraction: float = 0.1
    seed: int = 42
    # Bounds for each parameter  ``{name: (low, high)}``
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class GeneticAlgorithm:
    """Simple real-coded GA for structure-parameter optimisation.

    Usage
    -----
    >>> ga = GeneticAlgorithm(config, eval_fn)
    >>> best = ga.run()
    """

    def __init__(self, config: GAConfig, eval_fn: EvalFunc) -> None:
        self.cfg = config
        self.eval_fn = eval_fn
        self._rng = random.Random(config.seed)
        self._np_rng = np.random.default_rng(config.seed)
        self.population: List[Candidate] = []
        self.history: List[Candidate] = []

    # -- Initialisation ------------------------------------------------
    def _init_population(self) -> None:
        self.population = []
        for i in range(self.cfg.population_size):
            params: Dict[str, float] = {}
            for name, (lo, hi) in self.cfg.param_bounds.items():
                params[name] = self._rng.uniform(lo, hi)
            c = Candidate(params=params, label=f"gen0_{i}")
            c.fitness = self.eval_fn(params)
            c.generation = 0
            self.population.append(c)

    # -- Selection (tournament) ----------------------------------------
    def _select(self) -> Candidate:
        a, b = self._rng.sample(self.population, 2)
        return a if a.fitness <= b.fitness else b

    # -- Crossover (blend-alpha) ---------------------------------------
    def _crossover(self, p1: Candidate, p2: Candidate) -> Dict[str, float]:
        child: Dict[str, float] = {}
        alpha = 0.5
        for name in self.cfg.param_bounds:
            if self._rng.random() < self.cfg.crossover_rate:
                lo = min(p1.params[name], p2.params[name])
                hi = max(p1.params[name], p2.params[name])
                span = hi - lo
                child[name] = self._rng.uniform(lo - alpha * span,
                                                 hi + alpha * span)
            else:
                child[name] = p1.params[name]
            # Clamp to bounds
            blo, bhi = self.cfg.param_bounds[name]
            child[name] = max(blo, min(bhi, child[name]))
        return child

    # -- Mutation (Gaussian perturbation) ------------------------------
    def _mutate(self, params: Dict[str, float]) -> Dict[str, float]:
        mutated = dict(params)
        for name, (lo, hi) in self.cfg.param_bounds.items():
            if self._rng.random() < self.cfg.mutation_rate:
                sigma = (hi - lo) * 0.1
                mutated[name] += self._np_rng.normal(0, sigma)
                mutated[name] = max(lo, min(hi, mutated[name]))
        return mutated

    # -- Main loop -----------------------------------------------------
    def run(self) -> Candidate:
        """Execute the GA and return the best candidate found."""
        self._init_population()
        n_elite = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

        for gen in range(1, self.cfg.num_generations + 1):
            self.population.sort(key=lambda c: c.fitness)
            elites = self.population[:n_elite]
            self.history.extend(elites)

            new_pop = list(elites)
            while len(new_pop) < self.cfg.population_size:
                p1 = self._select()
                p2 = self._select()
                child_params = self._crossover(p1, p2)
                child_params = self._mutate(child_params)
                child = Candidate(
                    params=child_params,
                    label=f"gen{gen}_{len(new_pop)}",
                    generation=gen,
                )
                child.fitness = self.eval_fn(child_params)
                new_pop.append(child)

            self.population = new_pop
            best = min(self.population, key=lambda c: c.fitness)
            logger.info("GA gen %d/%d  best_fitness=%.6f",
                        gen, self.cfg.num_generations, best.fitness)

        self.population.sort(key=lambda c: c.fitness)
        return self.population[0]

    def top_candidates(self, n: int = 5) -> List[Candidate]:
        """Return the *n* best candidates from the final population."""
        return sorted(self.population, key=lambda c: c.fitness)[:n]
