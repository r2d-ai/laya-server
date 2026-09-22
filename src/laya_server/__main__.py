import uvicorn

from .settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "laya_server.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
