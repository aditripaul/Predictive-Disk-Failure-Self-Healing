"""Fleet state model: nodes, drives, replication groups, and the drive
state machine (docs/design_goal.md section 21).

    HEALTHY -> DEGRADED -> PREDICTED_FAILURE -> {CORDONED, MIGRATED, DRAINED}
                                                      -> REPLACED / RECOVERED
    (HEALTHY | DEGRADED | PREDICTED_FAILURE) -> FAILED
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class DriveState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PREDICTED_FAILURE = "PREDICTED_FAILURE"
    CORDONED = "CORDONED"
    MIGRATED = "MIGRATED"
    DRAINED = "DRAINED"
    REPLACED = "REPLACED"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


#: states that still count as "serving" capacity for quorum/last-node checks
HEALTHY_LIKE_STATES = {DriveState.HEALTHY, DriveState.DEGRADED, DriveState.PREDICTED_FAILURE}


@dataclass
class Drive:
    drive_id: str
    node_id: str
    replication_group_id: str
    state: DriveState = DriveState.HEALTHY


@dataclass
class Node:
    node_id: str
    failure_domain: str
    drive_ids: list[str] = field(default_factory=list)


@dataclass
class ReplicationGroup:
    group_id: str
    drive_ids: list[str]
    min_healthy_drives: int = 1


class FleetSimulator:
    def __init__(
        self,
        drives: list[Drive],
        nodes: list[Node],
        replication_groups: list[ReplicationGroup],
    ):
        self.drives: dict[str, Drive] = {d.drive_id: d for d in drives}
        self.nodes: dict[str, Node] = {n.node_id: n for n in nodes}
        self.replication_groups: dict[str, ReplicationGroup] = {
            g.group_id: g for g in replication_groups
        }
        self.stale_drive_ids: set[str] = set()

    def snapshot(self) -> dict:
        return {
            "fleet_snapshot_id": str(uuid.uuid4()),
            "drives": [
                {
                    "drive_id": d.drive_id,
                    "node_id": d.node_id,
                    "replication_group_id": d.replication_group_id,
                    "state": d.state.value,
                    "stale_telemetry": d.drive_id in self.stale_drive_ids,
                }
                for d in self.drives.values()
            ],
        }

    def is_last_healthy_node_in_domain(self, drive_id: str) -> bool:
        drive = self.drives[drive_id]
        node = self.nodes[drive.node_id]
        domain_nodes = [n for n in self.nodes.values() if n.failure_domain == node.failure_domain]
        healthy_nodes = [n for n in domain_nodes if self._node_is_healthy(n)]
        return self._node_is_healthy(node) and len(healthy_nodes) <= 1

    def quorum_ok_after_drain(self, drive_id: str) -> bool:
        drive = self.drives[drive_id]
        group = self.replication_groups.get(drive.replication_group_id)
        if group is None:
            return True
        healthy_after = sum(
            1
            for did in group.drive_ids
            if did != drive_id and self.drives[did].state in HEALTHY_LIKE_STATES
        )
        return healthy_after >= group.min_healthy_drives

    def _node_is_healthy(self, node: Node) -> bool:
        return any(self.drives[did].state in HEALTHY_LIKE_STATES for did in node.drive_ids)
