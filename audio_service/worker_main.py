import uvicorn

from .config import load_settings
from .worker_api import app


def main() -> None:
    settings = load_settings()
    uvicorn.run(app, host=settings.host, port=settings.worker_port, workers=1)


if __name__ == "__main__":
    main()
