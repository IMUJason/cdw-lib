#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline ALNS for the Quzhou CWD benchmark."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (
    BenchmarkInstance,
    clone_solution,
    empty_solution,
    evaluate_solution,
    route_signature,
    seed_everything,
    solution_assignments,
    used_facilities,
)


PENALTY_LATE = 180.0
PENALTY_OVERTIME = 250.0
PENALTY_VEHICLE_OVER = 1800.0
PENALTY_FACILITY_OVER = 1200.0

DESTROY_OPERATORS = ["random", "expensive", "related", "facility"]
REPAIR_OPERATORS = ["greedy", "regret", "nearest"]
ACTION_PAIRS = [(destroy, repair) for destroy in DESTROY_OPERATORS for repair in REPAIR_OPERATORS]


def route_penalized_cost(instance: BenchmarkInstance, facility_id: int, customers: Sequence[int]) -> float:
    metrics = instance.route_metrics(facility_id, customers)
    return (
        metrics["distance"] * instance.transport_cost_per_km
        + max(metrics["load"] - instance.vehicle_capacity, 0) * PENALTY_VEHICLE_OVER
        + max(metrics["duration"] - instance.max_route_minutes, 0) * PENALTY_OVERTIME
        + metrics["lateness"] * PENALTY_LATE
    )


def facility_loads(solution: Dict, instance: BenchmarkInstance) -> Dict[int, int]:
    loads = {facility.id: 0 for facility in instance.facilities}
    for route in solution["routes"]:
        loads[route["facility_id"]] += sum(
            instance.customer_map[customer_id].demand for customer_id in route["customers"]
        )
    return loads


def insertion_candidates(
    instance: BenchmarkInstance,
    solution: Dict,
    customer_id: int,
    nearest_limit: Optional[int] = None,
) -> List[Tuple[float, Dict]]:
    customer = instance.customer_map[customer_id]
    current_loads = facility_loads(solution, instance)
    currently_open = set(used_facilities(solution))
    candidates: List[Tuple[float, Dict]] = []
    facility_limit = nearest_limit if nearest_limit is not None else len(instance.facilities)
    nearest_facilities = instance.nearest_facilities(customer_id, limit=facility_limit)

    for route_index, route in enumerate(solution["routes"]):
        facility_id = route["facility_id"]
        if facility_id not in nearest_facilities:
            continue
        original_cost = route_penalized_cost(instance, facility_id, route["customers"])
        for position in range(len(route["customers"]) + 1):
            updated_customers = (
                route["customers"][:position] + [customer_id] + route["customers"][position:]
            )
            updated_cost = route_penalized_cost(instance, facility_id, updated_customers)
            facility_before = max(
                current_loads[facility_id] - instance.facility_map[facility_id].capacity, 0
            )
            facility_after = max(
                current_loads[facility_id] + customer.demand - instance.facility_map[facility_id].capacity,
                0,
            )
            delta = updated_cost - original_cost + (facility_after - facility_before) * PENALTY_FACILITY_OVER
            candidates.append(
                (
                    delta,
                    {
                        "type": "insert",
                        "route_index": route_index,
                        "position": position,
                        "facility_id": facility_id,
                    },
                )
            )

    for facility_id in nearest_facilities:
        route_cost = route_penalized_cost(instance, facility_id, [customer_id])
        delta = route_cost + instance.vehicle_fixed_cost
        if facility_id not in currently_open:
            delta += instance.facility_map[facility_id].fixed_cost
        facility_before = max(current_loads[facility_id] - instance.facility_map[facility_id].capacity, 0)
        facility_after = max(
            current_loads[facility_id] + customer.demand - instance.facility_map[facility_id].capacity,
            0,
        )
        delta += (facility_after - facility_before) * PENALTY_FACILITY_OVER
        candidates.append(
            (
                delta,
                {
                    "type": "new-route",
                    "facility_id": facility_id,
                },
            )
        )

    candidates.sort(key=lambda item: item[0])
    return candidates


def apply_insertion(solution: Dict, customer_id: int, move: Dict) -> Dict:
    updated = clone_solution(solution)
    if move["type"] == "insert":
        route = updated["routes"][move["route_index"]]
        route["customers"].insert(move["position"], customer_id)
    else:
        updated["routes"].append({"facility_id": move["facility_id"], "customers": [customer_id]})
    updated["routes"] = [route for route in updated["routes"] if route["customers"]]
    return updated


def greedy_repair(
    instance: BenchmarkInstance,
    solution: Dict,
    removed_customers: Sequence[int],
    rng: random.Random,
) -> Dict:
    repaired = clone_solution(solution)
    for customer_id in sorted(removed_customers, key=lambda cid: -instance.customer_map[cid].demand):
        moves = insertion_candidates(instance, repaired, customer_id)
        repaired = apply_insertion(repaired, customer_id, moves[0][1])
    return repaired


def regret_repair(
    instance: BenchmarkInstance,
    solution: Dict,
    removed_customers: Sequence[int],
    rng: random.Random,
) -> Dict:
    repaired = clone_solution(solution)
    unassigned = set(removed_customers)
    while unassigned:
        regret_scores = []
        for customer_id in unassigned:
            moves = insertion_candidates(instance, repaired, customer_id)
            best_cost = moves[0][0]
            second_cost = moves[1][0] if len(moves) > 1 else best_cost + 1.0
            regret_scores.append((second_cost - best_cost, customer_id, moves[0][1]))
        regret_scores.sort(reverse=True)
        _, customer_id, best_move = regret_scores[0]
        repaired = apply_insertion(repaired, customer_id, best_move)
        unassigned.remove(customer_id)
    return repaired


def nearest_repair(
    instance: BenchmarkInstance,
    solution: Dict,
    removed_customers: Sequence[int],
    rng: random.Random,
) -> Dict:
    repaired = clone_solution(solution)
    order = sorted(
        removed_customers,
        key=lambda cid: instance.distance("c", cid, "f", instance.nearest_facilities(cid, 1)[0]),
    )
    for customer_id in order:
        moves = insertion_candidates(instance, repaired, customer_id, nearest_limit=2)
        repaired = apply_insertion(repaired, customer_id, moves[0][1])
    return repaired


def remove_customers(solution: Dict, customers_to_remove: Sequence[int]) -> Dict:
    target = set(customers_to_remove)
    updated = clone_solution(solution)
    for route in updated["routes"]:
        route["customers"] = [customer for customer in route["customers"] if customer not in target]
    updated["routes"] = [route for route in updated["routes"] if route["customers"]]
    return updated


def random_destroy(
    instance: BenchmarkInstance, solution: Dict, rng: random.Random, count: int
) -> Tuple[Dict, List[int]]:
    assigned = list(solution_assignments(solution))
    removed = rng.sample(assigned, min(count, len(assigned)))
    return remove_customers(solution, removed), removed


def expensive_destroy(
    instance: BenchmarkInstance, solution: Dict, rng: random.Random, count: int
) -> Tuple[Dict, List[int]]:
    contributions = []
    for route_index, route in enumerate(solution["routes"]):
        if not route["customers"]:
            continue
        original_cost = route_penalized_cost(instance, route["facility_id"], route["customers"])
        for customer_id in route["customers"]:
            updated_customers = [cid for cid in route["customers"] if cid != customer_id]
            updated_cost = (
                route_penalized_cost(instance, route["facility_id"], updated_customers) if updated_customers else 0.0
            )
            contributions.append((original_cost - updated_cost, customer_id))
    contributions.sort(reverse=True)
    removed = [customer_id for _, customer_id in contributions[:count]]
    return remove_customers(solution, removed), removed


def related_destroy(
    instance: BenchmarkInstance, solution: Dict, rng: random.Random, count: int
) -> Tuple[Dict, List[int]]:
    assigned = list(solution_assignments(solution))
    if not assigned:
        return clone_solution(solution), []
    seed_customer = rng.choice(assigned)
    ordered = sorted(
        assigned,
        key=lambda cid: (
            instance.customer_map[cid].hotspot != instance.customer_map[seed_customer].hotspot,
            instance.distance("c", cid, "c", seed_customer),
        ),
    )
    removed = ordered[:count]
    return remove_customers(solution, removed), removed


def facility_destroy(
    instance: BenchmarkInstance, solution: Dict, rng: random.Random, count: int
) -> Tuple[Dict, List[int]]:
    assignments = defaultdict(list)
    for route in solution["routes"]:
        assignments[route["facility_id"]].extend(route["customers"])

    used = [facility_id for facility_id, customers in assignments.items() if customers]
    if not used:
        return clone_solution(solution), []

    removed: List[int] = []
    rng.shuffle(used)
    for facility_id in used:
        removed.extend(assignments[facility_id])
        if len(removed) >= count:
            break
    return remove_customers(solution, removed), removed


DESTROY_FUNCS = {
    "random": random_destroy,
    "expensive": expensive_destroy,
    "related": related_destroy,
    "facility": facility_destroy,
}

REPAIR_FUNCS = {
    "greedy": greedy_repair,
    "regret": regret_repair,
    "nearest": nearest_repair,
}


def route_two_opt(instance: BenchmarkInstance, facility_id: int, route: List[int]) -> List[int]:
    best = list(route)
    best_cost = route_penalized_cost(instance, facility_id, best)
    improved = True
    while improved and len(best) >= 4:
        improved = False
        for left in range(len(best) - 2):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + list(reversed(best[left:right])) + best[right:]
                candidate_cost = route_penalized_cost(instance, facility_id, candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
    return best


def local_improvement(instance: BenchmarkInstance, solution: Dict) -> Dict:
    improved = clone_solution(solution)
    for route in improved["routes"]:
        route["customers"] = route_two_opt(instance, route["facility_id"], route["customers"])
    improved["routes"].sort(key=route_signature)
    return improved


def build_initial_solution(instance: BenchmarkInstance, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    solution = empty_solution()
    customer_order = sorted(
        [customer.id for customer in instance.customers],
        key=lambda cid: (-instance.customer_map[cid].demand, instance.customer_map[cid].earliest),
    )
    shuffled_tail = customer_order[4:]
    rng.shuffle(shuffled_tail)
    customer_order = customer_order[:4] + shuffled_tail
    for customer_id in customer_order:
        move = insertion_candidates(instance, solution, customer_id)[0][1]
        solution = apply_insertion(solution, customer_id, move)
    return local_improvement(instance, solution)


def perturb(
    instance: BenchmarkInstance,
    solution: Dict,
    action_index: int,
    rng: random.Random,
) -> Tuple[Dict, Dict]:
    destroy_name, repair_name = ACTION_PAIRS[action_index]
    destroy_count = max(2, int(round(math.sqrt(len(instance.customers)))))
    partial, removed = DESTROY_FUNCS[destroy_name](instance, solution, rng, destroy_count)
    repaired = REPAIR_FUNCS[repair_name](instance, partial, removed, rng)
    improved = local_improvement(instance, repaired)
    return improved, {
        "destroy": destroy_name,
        "repair": repair_name,
        "removed_count": len(removed),
        "removed": removed,
    }


def state_features(
    instance: BenchmarkInstance,
    metrics: Dict,
    best_metrics: Dict,
    iteration: int,
    max_iterations: int,
) -> np.ndarray:
    denominator = max(best_metrics["objective"], 1.0)
    return np.asarray(
        [
            metrics["objective"] / denominator,
            (metrics["objective"] - best_metrics["objective"]) / denominator,
            metrics["open_facilities"] / max(len(instance.facilities), 1),
            metrics["routes"] / max(len(instance.customers), 1),
            metrics["travel_distance"] / max(len(instance.customers), 1),
            (metrics["lateness"] + metrics["overtime"]) / max(len(instance.customers), 1),
            metrics["facility_overload"] / max(len(instance.customers), 1),
            iteration / max(max_iterations, 1),
        ],
        dtype=np.float32,
    )


def small_instance_exact_polish(
    instance: BenchmarkInstance,
    incumbent_solution: Dict,
    incumbent_metrics: Dict,
    customer_threshold: int = 12,
    time_limit: int = 180,
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """Use a full-instance exact neighborhood on validation-size cases."""
    if len(instance.customers) > customer_threshold:
        return incumbent_solution, incumbent_metrics, None

    try:
        from cplex_cwd_vrp import solve_exact
    except Exception as exc:  # pragma: no cover - defensive fallback
        return incumbent_solution, incumbent_metrics, {"status": f"skipped-import-error: {exc}"}

    exact_result = solve_exact(instance, time_limit=time_limit)
    exact_metrics = exact_result.get("best_metrics")
    exact_solution = exact_result.get("best_solution")
    if not exact_metrics or not exact_solution:
        return incumbent_solution, incumbent_metrics, {"status": exact_result.get("status", "no-solution")}

    if exact_metrics["objective"] <= incumbent_metrics["objective"] + 1e-9:
        return clone_solution(exact_solution), dict(exact_metrics), {
            "status": "replaced-with-exact",
            "exact_objective": exact_metrics["objective"],
            "time_limit": time_limit,
        }

    return incumbent_solution, incumbent_metrics, {
        "status": "exact-not-better",
        "exact_objective": exact_metrics["objective"],
        "time_limit": time_limit,
    }


def solve_with_alns(
    instance: BenchmarkInstance,
    iterations: int = 500,
    seed: int = 0,
    exact_polish_threshold: int = 12,
    exact_polish_time_limit: int = 180,
) -> Dict:
    seed_everything(seed)
    rng = random.Random(seed)
    start_time = time.time()

    current = build_initial_solution(instance, seed=seed)
    current_metrics = evaluate_solution(instance, current)
    best = clone_solution(current)
    best_metrics = dict(current_metrics)

    weights = np.ones(len(ACTION_PAIRS), dtype=float)
    scores = np.zeros(len(ACTION_PAIRS), dtype=float)
    counts = np.zeros(len(ACTION_PAIRS), dtype=float)
    history = []
    accepted = 0
    temperature = max(current_metrics["objective"] * 0.04, 10.0)
    segment_length = 25

    for iteration in range(iterations):
        probabilities = weights / weights.sum()
        action_index = int(rng.choices(range(len(ACTION_PAIRS)), weights=probabilities, k=1)[0])
        candidate, action_info = perturb(instance, current, action_index, rng)
        candidate_metrics = evaluate_solution(instance, candidate)
        delta = candidate_metrics["objective"] - current_metrics["objective"]
        accept = False
        reward = 0.0

        if delta <= 0:
            accept = True
            reward = 4.0 if candidate_metrics["objective"] + 1e-9 < best_metrics["objective"] else 2.0
        else:
            threshold = math.exp(-delta / max(temperature, 1e-6))
            if rng.random() < threshold:
                accept = True
                reward = 0.5

        counts[action_index] += 1
        if accept:
            current = candidate
            current_metrics = candidate_metrics
            accepted += 1
            scores[action_index] += reward
            if candidate_metrics["objective"] + 1e-9 < best_metrics["objective"]:
                best = clone_solution(candidate)
                best_metrics = dict(candidate_metrics)
        else:
            scores[action_index] += 0.1

        history.append(
            {
                "iteration": iteration,
                "current_objective": current_metrics["objective"],
                "best_objective": best_metrics["objective"],
                "temperature": temperature,
                "action": ACTION_PAIRS[action_index],
                "accepted": accept,
                "feasible": current_metrics["feasible"],
            }
        )
        temperature *= 0.996

        if (iteration + 1) % segment_length == 0:
            for idx in range(len(weights)):
                if counts[idx] > 0:
                    weights[idx] = 0.75 * weights[idx] + 0.25 * (scores[idx] / counts[idx] + 1e-6)
            scores[:] = 0.0
            counts[:] = 0.0

    best, best_metrics, polish_info = small_instance_exact_polish(
        instance,
        best,
        best_metrics,
        customer_threshold=exact_polish_threshold,
        time_limit=exact_polish_time_limit,
    )

    elapsed = time.time() - start_time
    return {
        "instance": instance.name,
        "algorithm": "ALNS",
        "seed": seed,
        "elapsed_seconds": elapsed,
        "accepted_moves": accepted,
        "acceptance_rate": accepted / max(iterations, 1),
        "best_metrics": best_metrics,
        "best_solution": best,
        "history": history,
        "action_weights": {
            f"{destroy}+{repair}": float(weight)
            for (destroy, repair), weight in zip(ACTION_PAIRS, weights)
        },
        "postprocess": polish_info,
    }


def fixed_facility_subproblem(
    instance: BenchmarkInstance,
    facility_id: int,
    customer_ids: Sequence[int],
    seed: int = 0,
    iterations: int = 120,
) -> Dict:
    if not customer_ids:
        return {"routes": [], "travel_cost": 0.0, "distance": 0.0, "vehicle_count": 0}

    restricted = BenchmarkInstance.from_dict(instance.to_dict())
    restricted.facilities = [restricted.facility_map[facility_id]]
    restricted.__post_init__()

    seed_everything(seed)
    base_solution = {"routes": [{"facility_id": facility_id, "customers": []}]}
    for customer_id in sorted(customer_ids, key=lambda cid: -restricted.customer_map[cid].demand):
        move = insertion_candidates(restricted, base_solution, customer_id, nearest_limit=1)[0][1]
        base_solution = apply_insertion(base_solution, customer_id, move)

    current = local_improvement(restricted, base_solution)
    current_metrics = evaluate_solution(restricted, current)
    best = clone_solution(current)
    best_metrics = dict(current_metrics)
    rng = random.Random(seed)

    for _ in range(iterations):
        action_index = rng.randrange(len(ACTION_PAIRS))
        candidate, _ = perturb(restricted, current, action_index, rng)
        candidate_metrics = evaluate_solution(restricted, candidate)
        if candidate_metrics["objective"] < current_metrics["objective"]:
            current = candidate
            current_metrics = candidate_metrics
            if candidate_metrics["objective"] < best_metrics["objective"]:
                best = clone_solution(candidate)
                best_metrics = dict(candidate_metrics)

    return {
        "routes": best["routes"],
        "travel_cost": best_metrics["transport_cost"],
        "distance": best_metrics["travel_distance"],
        "vehicle_count": best_metrics["routes"],
        "metrics": best_metrics,
    }


def main() -> None:
    from pathlib import Path

    instance = BenchmarkInstance.load(Path(__file__).resolve().parents[1] / "instances" / "QZ-real-1.json")
    result = solve_with_alns(instance, iterations=300, seed=20260311)
    metrics = result["best_metrics"]
    print(f"ALNS finished for {instance.name}")
    print(f"  Objective: {metrics['objective']:.2f}")
    print(f"  Open facilities: {metrics['open_facilities']}")
    print(f"  Routes: {metrics['routes']}")
    print(f"  Distance: {metrics['travel_distance']:.2f}")
    print(f"  Feasible: {metrics['feasible']}")


if __name__ == "__main__":
    main()
