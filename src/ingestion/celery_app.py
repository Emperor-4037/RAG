from celery import Celery
from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,  # CPU heavy tasks, don't prefetch heavily
    task_acks_late=True,
)

# Import tasks to register them with the Celery app
import src.ingestion.tasks  # noqa
