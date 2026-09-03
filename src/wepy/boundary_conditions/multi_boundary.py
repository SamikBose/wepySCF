"""Generic Boolean combinations of interatomic-distance boundaries.

This file is staged in the CPD workspace before installation into wepy_dev.
"""

from collections import defaultdict

import numpy as np

from wepy.boundary_conditions.boundary import WarpBC


class MultiBoundaryBC(WarpBC):
    """Warp walkers that satisfy any named Boolean boundary definition.

    A boundary definition has a ``name``, an integer ``warp_class``, and a
    nested ``condition``. Condition nodes use ``logic`` equal to ``all``,
    ``any``, or ``not``; leaf nodes contain ``pair``, ``comparison`` and
    ``cutoff``. Supported comparisons are ``lt``, ``le``, ``gt`` and ``ge``.

    Example::

        {"name": "product", "warp_class": 1,
         "condition": {"logic": "all", "conditions": [
             {"pair": (0, 4), "comparison": "le", "cutoff": 3.2},
             {"logic": "any", "conditions": [
                 {"pair": (1, 5), "comparison": "le", "cutoff": 3.2},
                 {"pair": (2, 6), "comparison": "le", "cutoff": 3.2},
             ]}]}}

    The top-level definitions are implicitly ORed. ``boundary_mask`` stores
    every matched definition as a bit mask. ``boundary_id`` and ``warp_class``
    use the first matched definition, making definition order the explicit
    priority rule if overlapping definitions have different classes.
    """

    WARPING_FIELDS = WarpBC.WARPING_FIELDS + (
        "boundary_mask",
        "boundary_id",
        "warp_class",
    )
    WARPING_SHAPES = WarpBC.WARPING_SHAPES + ((1,), (1,), (1,))
    WARPING_DTYPES = WarpBC.WARPING_DTYPES + (np.uint64, int, int)
    WARPING_RECORD_FIELDS = WARPING_FIELDS

    PROGRESS_FIELDS = ("distances", "boundary_mask", "boundary_id", "warp_class")
    PROGRESS_SHAPES = (Ellipsis, (1,), (1,), (1,))
    PROGRESS_DTYPES = (float, np.uint64, int, int)
    PROGRESS_RECORD_FIELDS = PROGRESS_FIELDS

    _COMPARISONS = {
        "lt": np.less,
        "le": np.less_equal,
        "gt": np.greater,
        "ge": np.greater_equal,
    }

    def __init__(self, *args, boundary_definitions, tracked_pairs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not boundary_definitions:
            raise ValueError("boundary_definitions must contain at least one definition")
        if len(boundary_definitions) > 64:
            raise ValueError("boundary_mask supports at most 64 definitions")

        self._boundary_definitions = tuple(boundary_definitions)
        names = [definition.get("name") for definition in self._boundary_definitions]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every boundary definition requires a nonempty name")
        if len(set(names)) != len(names):
            raise ValueError("boundary definition names must be unique")
        for definition in self._boundary_definitions:
            warp_class = definition.get("warp_class")
            if not isinstance(warp_class, (int, np.integer)) or int(warp_class) <= 0:
                raise ValueError("every boundary definition requires a positive integer warp_class")
            self._validate_condition(definition.get("condition"))

        condition_pairs = []
        for definition in self._boundary_definitions:
            condition_pairs.extend(self._condition_pairs(definition["condition"]))
        ordered_pairs = condition_pairs if tracked_pairs is None else list(tracked_pairs)
        self._tracked_pairs = tuple(dict.fromkeys(tuple(pair) for pair in ordered_pairs))
        missing = set(condition_pairs).difference(self._tracked_pairs)
        if missing:
            raise ValueError(f"tracked_pairs omits condition pairs: {sorted(missing)}")
        self._pair_indices = {pair: idx for idx, pair in enumerate(self._tracked_pairs)}

        self.PROGRESS_SHAPES = (
            (len(self._tracked_pairs),),
            (1,),
            (1,),
            (1,),
        )

    @property
    def boundary_definitions(self):
        return self._boundary_definitions

    @property
    def tracked_pairs(self):
        return self._tracked_pairs

    @classmethod
    def _validate_condition(cls, condition):
        if not isinstance(condition, dict):
            raise ValueError("condition must be a dictionary")
        if "logic" in condition:
            logic = condition["logic"]
            children = condition.get("conditions")
            if logic not in ("all", "any", "not"):
                raise ValueError(f"unsupported Boolean logic: {logic!r}")
            if not isinstance(children, (list, tuple)) or not children:
                raise ValueError(f"{logic!r} requires nonempty conditions")
            if logic == "not" and len(children) != 1:
                raise ValueError("'not' requires exactly one child condition")
            for child in children:
                cls._validate_condition(child)
            return
        pair = condition.get("pair")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("distance leaf requires a two-atom pair")
        if condition.get("comparison") not in cls._COMPARISONS:
            raise ValueError("comparison must be one of lt, le, gt, ge")
        cutoff = condition.get("cutoff")
        if not isinstance(cutoff, (int, float, np.number)) or not np.isfinite(cutoff):
            raise ValueError("distance leaf requires a finite numeric cutoff")

    @classmethod
    def _condition_pairs(cls, condition):
        if "logic" in condition:
            pairs = []
            for child in condition["conditions"]:
                pairs.extend(cls._condition_pairs(child))
            return pairs
        return [tuple(int(idx) for idx in condition["pair"])]

    def _distances(self, walker):
        positions = np.asarray(walker.state["positions"], dtype=float)
        return np.asarray(
            [np.linalg.norm(positions[i] - positions[j]) for i, j in self._tracked_pairs],
            dtype=float,
        )

    def _evaluate_condition(self, condition, distances):
        if "logic" in condition:
            values = [self._evaluate_condition(child, distances) for child in condition["conditions"]]
            if condition["logic"] == "all":
                return all(values)
            if condition["logic"] == "any":
                return any(values)
            return not values[0]
        pair = tuple(int(idx) for idx in condition["pair"])
        distance = distances[self._pair_indices[pair]]
        comparison = self._COMPARISONS[condition["comparison"]]
        return bool(comparison(distance, float(condition["cutoff"])))

    def _progress(self, walker):
        distances = self._distances(walker)
        matched = [
            self._evaluate_condition(definition["condition"], distances)
            for definition in self._boundary_definitions
        ]
        boundary_mask = sum((1 << idx) for idx, value in enumerate(matched) if value)
        first_idx = next((idx for idx, value in enumerate(matched) if value), None)
        boundary_id = 0 if first_idx is None else first_idx + 1
        warp_class = (
            0
            if first_idx is None
            else int(self._boundary_definitions[first_idx]["warp_class"])
        )
        return boundary_mask != 0, {
            "distances": distances,
            "boundary_mask": np.asarray([boundary_mask], dtype=np.uint64),
            "boundary_id": np.asarray([boundary_id], dtype=int),
            "warp_class": np.asarray([warp_class], dtype=int),
        }

    def warp_walkers(self, walkers, cycle):
        """Warp walkers while copying boundary classification into warp records."""
        new_walkers = []
        warp_data = []
        progress_data = defaultdict(list)
        all_progress_data = [self._progress(walker) for walker in walkers]

        for walker_idx, walker in enumerate(walkers):
            to_warp, walker_progress = all_progress_data[walker_idx]
            for key, value in walker_progress.items():
                progress_data[key].append(value)
            if to_warp:
                warped_walker, record = self._warp(walker)
                record["walker_idx"] = np.asarray([walker_idx], dtype=int)
                for field in ("boundary_mask", "boundary_id", "warp_class"):
                    record[field] = np.asarray(walker_progress[field]).reshape(1)
                new_walkers.append(warped_walker)
                warp_data.append(record)
            else:
                new_walkers.append(walker)

        bc_data = self._update_bc(new_walkers, warp_data, progress_data, cycle)
        return new_walkers, warp_data, bc_data, progress_data
