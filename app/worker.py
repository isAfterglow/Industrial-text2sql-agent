"""RQ worker entrypoint: python -m app.worker"""
from redis import Redis
from rq import Queue, Worker
from app.config import get_settings
def main() -> None:
    settings = get_settings()
    if not settings.REDIS_URL: raise RuntimeError("REDIS_URL is required for the RQ worker")
    Worker([Queue(settings.TASK_QUEUE_NAME, connection=Redis.from_url(settings.REDIS_URL, protocol=2))]).work()
if __name__ == "__main__": main()
