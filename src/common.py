#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data structures and evaluation utilities for Plan A+."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLAN_A_ROOT = ROOT.parent
RAW_ROOT = PLAN_A_ROOT / "raw"
V2_ROOT = PLAN_A_ROOT / "v2"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(a))


def km_to_lon_delta(km: float, latitude: float) -> float:
    return km / (111.320 * max(math.cos(math.radians(latitude)), 0.25))


def km_to_lat_delta(km: float) -> float:
    return km / 110.574


@dataclass
class Customer:
    id: int
    name: str
    x: float
    y: float
    demand: int
    earliest: int
    latest: int
    service_time: int
    hotspot: str
    hotspot_weight: float
    source_project: str


@dataclass
class Facility:
    id: int
    name: str
    x: float
    y: float
    capacity: int
    fixed_cost: float
    source_site: str
    site_weight: float


@dataclass
class BenchmarkInstance:
    name: str
    description: str
    horizon_label: str
    vehicle_capacity: int
    vehicle_fixed_cost: float
    transport_cost_per_km: float
    max_route_minutes: int
    day_start: int
    day_end: int
    average_speed_kmph: float
    restricted_center: Dict[str, float]
    restricted_radius_km: float
    restricted_window: Tuple[int, int]
    restricted_penalty_factor: float
    customers: List[Customer] = field(default_factory=list)
    facilities: List[Facility] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.customer_map = {customer.id: customer for customer in self.customers}
        self.facility_map = {facility.id: facility for facility in self.facilities}
        self._distance_cache: Dict[Tuple[str, int, str, int], float] = {}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "horizon_label": self.horizon_label,
            "vehicle_capacity": self.vehicle_capacity,
            "vehicle_fixed_cost": self.vehicle_fixed_cost,
            "transport_cost_per_km": self.transport_cost_per_km,
            "max_route_minutes": self.max_route_minutes,
            "day_start": self.day_start,
            "day_end": self.day_end,
            "average_speed_kmph": self.average_speed_kmph,
            "restricted_center": self.restricted_center,
            "restricted_radius_km": self.restricted_radius_km,
            "restricted_window": list(self.restricted_window),
            "restricted_penalty_factor": self.restricted_penalty_factor,
            "customers": [asdict(customer) for customer in self.customers],
            "facilities": [asdict(facility) for facility in self.facilities],
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "BenchmarkInstance":
        return cls(
            name=payload["name"],
            description=payload["description"],
            horizon_label=payload["horizon_label"],
            vehicle_capacity=int(payload["vehicle_capacity"]),
            vehicle_fixed_cost=float(payload["vehicle_fixed_cost"]),
            transport_cost_per_km=float(payload["transport_cost_per_km"]),
            max_route_minutes=int(payload["max_route_minutes"]),
            day_start=int(payload["day_start"]),
            day_end=int(payload["day_end"]),
            average_speed_kmph=float(payload["average_speed_kmph"]),
            restricted_center=payload["restricted_center"],
            restricted_radius_km=float(payload["restricted_radius_km"]),
            restricted_window=tuple(payload["restricted_window"]),
            restricted_penalty_factor=float(payload["restricted_penalty_factor"]),
            customers=[Customer(**customer) for customer in payload["customers"]],
            facilities=[Facility(**facility) for facility in payload["facilities"]],
        )

    @classmethod
    def load(cls, path: Path) -> "BenchmarkInstance":
        return cls.from_dict(read_json(path))

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    def distance(
        self, left_kind: str, left_id: int, right_kind: str, right_id: int
    ) -> float:
        key = (left_kind, left_id, right_kind, right_id)
        if key in self._distance_cache:
            return self._distance_cache[key]

        if left_kind == "c":
            left = self.customer_map[left_id]
        else:
            left = self.facility_map[left_id]
        if right_kind == "c":
            right = self.customer_map[right_id]
        else:
            right = self.facility_map[right_id]

        value = haversine_km(left.x, left.y, right.x, right.y)
        self._distance_cache[key] = value
        self._distance_cache[(right_kind, right_id, left_kind, left_id)] = value
        return value

    def travel_time(
        self,
        left_kind: str,
        left_id: int,
        right_kind: str,
        right_id: int,
        departure_time: float,
    ) -> float:
        distance_km = self.distance(left_kind, left_id, right_kind, right_id)
        base_minutes = distance_km / max(self.average_speed_kmph, 1e-6) * 60.0

        if left_kind == "c":
            left = self.customer_map[left_id]
        else:
            left = self.facility_map[left_id]
        if right_kind == "c":
            right = self.customer_map[right_id]
        else:
            right = self.facility_map[right_id]

        midpoint_lon = (left.x + right.x) / 2.0
        midpoint_lat = (left.y + right.y) / 2.0
        midpoint_distance = haversine_km(
            midpoint_lon,
            midpoint_lat,
            self.restricted_center["x"],
            self.restricted_center["y"],
        )
        is_restricted = midpoint_distance <= self.restricted_radius_km
        in_window = self.restricted_window[0] <= departure_time <= self.restricted_window[1]
        if is_restricted and in_window:
            base_minutes *= self.restricted_penalty_factor

        return base_minutes

    def route_metrics(self, facility_id: int, customers: Sequence[int]) -> Dict[str, float]:
        facility = self.facility_map[facility_id]
        current_time = float(self.day_start)
        total_distance = 0.0
        total_wait = 0.0
        total_lateness = 0.0
        total_load = 0
        previous_kind = "f"
        previous_id = facility.id

        for customer_id in customers:
            customer = self.customer_map[customer_id]
            travel_minutes = self.travel_time(
                previous_kind, previous_id, "c", customer.id, current_time
            )
            total_distance += self.distance(previous_kind, previous_id, "c", customer.id)
            current_time += travel_minutes
            if current_time < customer.earliest:
                total_wait += customer.earliest - current_time
                current_time = float(customer.earliest)
            if current_time > customer.latest:
                total_lateness += current_time - customer.latest
            current_time += customer.service_time
            total_load += customer.demand
            previous_kind = "c"
            previous_id = customer.id

        if customers:
            travel_minutes = self.travel_time(previous_kind, previous_id, "f", facility.id, current_time)
            total_distance += self.distance(previous_kind, previous_id, "f", facility.id)
            current_time += travel_minutes

        duration = current_time - self.day_start
        return {
            "distance": total_distance,
            "duration": duration,
            "lateness": total_lateness,
            "wait": total_wait,
            "load": total_load,
            "capacity_slack": self.vehicle_capacity - total_load,
            "time_slack": self.max_route_minutes - duration,
        }

    def nearest_facilities(self, customer_id: int, limit: int | None = None) -> List[int]:
        ordering = sorted(
            self.facility_map,
            key=lambda facility_id: self.distance("c", customer_id, "f", facility_id),
        )
        return ordering if limit is None else ordering[:limit]


def empty_solution() -> Dict:
    return {"routes": []}


def clone_solution(solution: Dict) -> Dict:
    return {
        "routes": [
            {"facility_id": route["facility_id"], "customers": list(route["customers"])}
            for route in solution["routes"]
        ]
    }


def solution_assignments(solution: Dict) -> Dict[int, int]:
    assignments = {}
    for route in solution["routes"]:
        for customer_id in route["customers"]:
            assignments[customer_id] = route["facility_id"]
    return assignments


def used_facilities(solution: Dict) -> List[int]:
    return sorted({route["facility_id"] for route in solution["routes"] if route["customers"]})


def missing_customers(instance: BenchmarkInstance, solution: Dict) -> List[int]:
    assigned = solution_assignments(solution)
    return [customer.id for customer in instance.customers if customer.id not in assigned]


def duplicate_customers(solution: Dict) -> List[int]:
    seen = set()
    duplicates = set()
    for route in solution["routes"]:
        for customer_id in route["customers"]:
            if customer_id in seen:
                duplicates.add(customer_id)
            seen.add(customer_id)
    return sorted(duplicates)


def evaluate_solution(instance: BenchmarkInstance, solution: Dict) -> Dict[str, float]:
    fixed_cost = 0.0
    vehicle_cost = 0.0
    transport_cost = 0.0
    travel_distance = 0.0
    lateness = 0.0
    overload = 0.0
    overtime = 0.0
    facility_usage = {facility.id: 0 for facility in instance.facilities}
    route_records: List[Dict] = []

    for route in solution["routes"]:
        if not route["customers"]:
            continue
        metrics = instance.route_metrics(route["facility_id"], route["customers"])
        facility_usage[route["facility_id"]] += metrics["load"]
        vehicle_cost += instance.vehicle_fixed_cost
        transport_cost += metrics["distance"] * instance.transport_cost_per_km
        travel_distance += metrics["distance"]
        lateness += metrics["lateness"]
        overload += max(metrics["load"] - instance.vehicle_capacity, 0)
        overtime += max(metrics["duration"] - instance.max_route_minutes, 0)
        route_records.append(
            {
                "facility_id": route["facility_id"],
                "customers": list(route["customers"]),
                **metrics,
            }
        )

    for facility_id, load in facility_usage.items():
        if load > 0:
            fixed_cost += instance.facility_map[facility_id].fixed_cost

    missing = missing_customers(instance, solution)
    duplicates = duplicate_customers(solution)
    facility_overload = sum(
        max(facility_usage[facility_id] - instance.facility_map[facility_id].capacity, 0)
        for facility_id in facility_usage
    )

    penalty = (
        10000.0 * len(missing)
        + 8000.0 * len(duplicates)
        + 1800.0 * overload
        + 1200.0 * facility_overload
        + 250.0 * overtime
        + 180.0 * lateness
    )
    objective = fixed_cost + vehicle_cost + transport_cost + penalty
    feasible = penalty < 1e-6

    return {
        "objective": objective,
        "fixed_cost": fixed_cost,
        "vehicle_cost": vehicle_cost,
        "transport_cost": transport_cost,
        "travel_distance": travel_distance,
        "penalty": penalty,
        "lateness": lateness,
        "vehicle_overload": overload,
        "facility_overload": facility_overload,
        "overtime": overtime,
        "missing_count": len(missing),
        "duplicate_count": len(duplicates),
        "open_facilities": len([f for f, load in facility_usage.items() if load > 0]),
        "routes": len(route_records),
        "feasible": feasible,
        "route_records": route_records,
        "missing_customers": missing,
        "duplicate_customers": duplicates,
        "facility_usage": facility_usage,
    }


def format_minutes(minutes: float) -> str:
    total = int(round(minutes))
    hours, remainder = divmod(total, 60)
    return f"{hours:02d}:{remainder:02d}"


def route_signature(route: Dict) -> Tuple[int, Tuple[int, ...]]:
    return route["facility_id"], tuple(route["customers"])


def sorted_customer_ids(instance: BenchmarkInstance) -> List[int]:
    return [customer.id for customer in sorted(instance.customers, key=lambda customer: customer.id)]


def weighted_quantile(values: Iterable[float], quantile: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), quantile))
