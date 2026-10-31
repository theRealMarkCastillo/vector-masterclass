"""Bounded operational probes for this course's disposable collections."""

import re
import subprocess
import time

from qdrant_client import models as qm


def wait_index(client, name, count, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        if (info.indexed_vectors_count or 0) >= int(count * 0.99) and str(info.status) == "green":
            return info
        time.sleep(0.25)
    raise TimeoutError(f"Index readiness gate not reached for {name}: {info}")


def collection_storage_bytes(name):
    if not re.fullmatch(r"vmc_[a-zA-Z0-9_]+", name):
        raise ValueError("Only course collection paths may be inspected")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "qdrant",
            "du",
            "-sk",
            f"/qdrant/storage/collections/{name}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return int(result.stdout.split()[0]) * 1024


def all_points(client, name):
    points, offset = [], None
    while True:
        page, offset = client.scroll(
            name, limit=256, offset=offset, with_payload=True, with_vectors=True
        )
        points.extend(page)
        if offset is None:
            return sorted(points, key=lambda point: str(point.id))


def switch_alias(client, alias, target):
    """One Qdrant alias transaction; does not atomically change external policy."""
    current = {item.alias_name for item in client.get_aliases().aliases}
    operations = []
    if alias in current:
        operations.append(qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias)))
    operations.append(
        qm.CreateAliasOperation(
            create_alias=qm.CreateAlias(collection_name=target, alias_name=alias)
        )
    )
    client.update_collection_aliases(operations)
