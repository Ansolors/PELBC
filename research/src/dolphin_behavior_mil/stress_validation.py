"""Immutable partition access for registered temporal, location and recorder stress tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .modeling_data import DatasetRegistry, read_jsonl


FORWARD_SCENARIO_ID = "forward_train_2021_2023_validate_2024_test_2025"
STRESS_INNER_FOLDS = tuple(f"stress_inner_{index:02d}" for index in range(1, 5))


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    buffer_ids: tuple[str, ...]

    @property
    def uses_fixed_forward_validation(self) -> bool:
        return self.scenario_id == FORWARD_SCENARIO_ID


@dataclass(frozen=True)
class StressScenarioRegistry:
    dataset: DatasetRegistry
    scenarios: Mapping[str, StressScenario]
    inner_fold_by_scenario_encounter: Mapping[str, Mapping[str, str]]

    @classmethod
    def load(cls, project_root: Path) -> "StressScenarioRegistry":
        root = Path(project_root).resolve()
        dataset = DatasetRegistry.load(root)
        version = root / "data/processed/v0.4.0"
        assignment_rows = read_jsonl(version / "temporal_stress_assignments.jsonl")
        assignment_rows.extend(
            read_jsonl(version / "location_stress_assignments.jsonl")
        )
        rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
        for row in assignment_rows:
            rows_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
        scenarios: dict[str, StressScenario] = {}
        expected = set(dataset.encounter_ids)
        for scenario_id, rows in sorted(rows_by_scenario.items()):
            ids = [str(row["encounter_id"]) for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError(f"stress scenario does not cover the cohort: {scenario_id}")
            by_role: dict[str, list[str]] = {}
            for row in rows:
                by_role.setdefault(str(row["role"]), []).append(str(row["encounter_id"]))
            known_roles = {"train_pool", "validation", "test", "buffer_excluded"}
            if set(by_role) - known_roles or not by_role.get("train_pool") or not by_role.get("test"):
                raise ValueError(f"invalid stress roles: {scenario_id}")
            scenario = StressScenario(
                scenario_id=scenario_id,
                train_ids=tuple(sorted(by_role.get("train_pool", ()))),
                validation_ids=tuple(sorted(by_role.get("validation", ()))),
                test_ids=tuple(sorted(by_role.get("test", ()))),
                buffer_ids=tuple(sorted(by_role.get("buffer_excluded", ()))),
            )
            if scenario.uses_fixed_forward_validation != bool(scenario.validation_ids):
                raise ValueError(f"unexpected fixed validation role: {scenario_id}")
            dataset.assert_campaign_disjoint(scenario.train_ids, scenario.test_ids)
            if scenario.validation_ids:
                dataset.assert_campaign_disjoint(
                    scenario.train_ids, scenario.validation_ids
                )
                dataset.assert_campaign_disjoint(
                    scenario.validation_ids, scenario.test_ids
                )
            scenarios[scenario_id] = scenario

        inner_rows = read_jsonl(version / "stress_inner_fold_assignments.jsonl")
        inner: dict[str, dict[str, str]] = {}
        for row in inner_rows:
            scenario_id = str(row["scenario_id"])
            encounter_id = str(row["encounter_id"])
            fold_id = str(row["inner_validation_fold_id"])
            values = inner.setdefault(scenario_id, {})
            if encounter_id in values:
                raise ValueError(
                    f"duplicate stress inner assignment: {scenario_id}/{encounter_id}"
                )
            values[encounter_id] = fold_id
        expected_inner_scenarios = set(scenarios) - {FORWARD_SCENARIO_ID}
        if set(inner) != expected_inner_scenarios:
            raise ValueError("stress inner scenario registry is incomplete")
        for scenario_id, values in inner.items():
            scenario = scenarios[scenario_id]
            if set(values) != set(scenario.train_ids):
                raise ValueError(f"stress inner rows do not cover train pool: {scenario_id}")
            if set(values.values()) != set(STRESS_INNER_FOLDS):
                raise ValueError(f"stress inner fold coverage mismatch: {scenario_id}")
            for fold_id in STRESS_INNER_FOLDS:
                validation_ids = tuple(
                    sorted(key for key, value in values.items() if value == fold_id)
                )
                train_ids = tuple(sorted(set(scenario.train_ids) - set(validation_ids)))
                dataset.assert_campaign_disjoint(train_ids, validation_ids)
        return cls(
            dataset=dataset,
            scenarios=scenarios,
            inner_fold_by_scenario_encounter=inner,
        )

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenarios))

    def scenario(self, scenario_id: str) -> StressScenario:
        key = str(scenario_id)
        if key not in self.scenarios:
            raise KeyError(f"unknown stress scenario: {key}")
        return self.scenarios[key]

    def selection_fold_ids(self, scenario_id: str) -> tuple[str, ...]:
        scenario = self.scenario(scenario_id)
        if scenario.uses_fixed_forward_validation:
            return ("fixed_validation_2024",)
        return STRESS_INNER_FOLDS

    def selection_split(
        self,
        scenario_id: str,
        fold_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        scenario = self.scenario(scenario_id)
        requested = str(fold_id)
        if scenario.uses_fixed_forward_validation:
            if requested != "fixed_validation_2024":
                raise KeyError(requested)
            return scenario.train_ids, scenario.validation_ids
        if requested not in STRESS_INNER_FOLDS:
            raise KeyError(requested)
        assignments = self.inner_fold_by_scenario_encounter[scenario.scenario_id]
        validation_ids = tuple(
            sorted(key for key, value in assignments.items() if value == requested)
        )
        train_ids = tuple(sorted(set(scenario.train_ids) - set(validation_ids)))
        self.dataset.assert_campaign_disjoint(train_ids, validation_ids)
        return train_ids, validation_ids
