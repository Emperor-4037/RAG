"""
Qdrant collection snapshot automation utility.
Creates a snapshot of the document collection and saves it locally.
Can be invoked directly or wired to a scheduler.
"""
import time
import datetime
import structlog
import httpx
from pathlib import Path
from src.config import get_settings

logger = structlog.get_logger(__name__)
BACKUP_DIR = Path("data/backups")


async def create_snapshot() -> dict:
    """
    Trigger a Qdrant snapshot of the active document collection.
    Snapshots are saved locally by Qdrant and can be downloaded.
    """
    settings = get_settings()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    collection = settings.qdrant_collection_name
    snapshot_url = f"{settings.qdrant_url}/collections/{collection}/snapshots"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Create snapshot (POST triggers Qdrant to generate the snapshot file)
        response = await client.post(snapshot_url)
        response.raise_for_status()
        snapshot_info = response.result().json().get("result", {})
        snapshot_name = snapshot_info.get("name")

        if not snapshot_name:
            raise ValueError("Qdrant did not return a snapshot name.")

        # 2. Download snapshot
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = BACKUP_DIR / f"{collection}_{timestamp}.snapshot"

        download_url = f"{snapshot_url}/{snapshot_name}"
        download_resp = await client.get(download_url)
        download_resp.raise_for_status()

        with open(dest, "wb") as f:
            f.write(download_resp.content)

        logger.info(
            "backup.snapshot_created",
            collection=collection,
            snapshot=snapshot_name,
            dest=str(dest),
            size_bytes=dest.stat().st_size,
        )

        return {
            "collection": collection,
            "snapshot": snapshot_name,
            "saved_to": str(dest),
            "size_bytes": dest.stat().st_size,
        }


async def list_snapshots() -> list:
    """List all snapshots Qdrant has recorded for the document collection."""
    settings = get_settings()
    collection = settings.qdrant_collection_name
    snapshot_url = f"{settings.qdrant_url}/collections/{collection}/snapshots"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(snapshot_url)
        response.raise_for_status()
        return response.json().get("result", [])


if __name__ == "__main__":
    import asyncio
    asyncio.run(create_snapshot())
