"""验证世界模型状态类迁移后的导入与行为兼容性。"""

from __future__ import annotations

import numpy as np

from mci_world_model.sdk._world_model import (
    CausalWorldModelState as ImportedCausalWorldModelState,
)
from mci_world_model.sdk._world_model import TrajectoryStep as ImportedTrajectoryStep
from mci_world_model.sdk._world_model import WorkingMemory as ImportedWorkingMemory
from mci_world_model.sdk._world_model_state import CausalWorldModelState, TrajectoryStep, WorkingMemory


def test_legacy_state_import_paths_re_export_same_classes() -> None:
    assert ImportedCausalWorldModelState is CausalWorldModelState
    assert ImportedTrajectoryStep is TrajectoryStep
    assert ImportedWorkingMemory is WorkingMemory


def test_empty_state_keeps_matrix_shapes() -> None:
    state = CausalWorldModelState.empty()

    assert state.to_adjacency_matrix().shape == (0, 0)
    assert state.to_node_feature_matrix().shape == (0, 8)


def test_working_memory_trajectory_behavior_unchanged() -> None:
    memory = WorkingMemory(max_length=1)
    memory.push(TrajectoryStep(state=CausalWorldModelState.empty(), step_index=0))

    assert len(memory.trajectory) == 1
    assert memory.get_recent(3)[0].step_index == 0
    assert np.isfinite(memory.trajectory[0].timestamp)
