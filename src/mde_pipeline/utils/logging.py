import logging
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,   
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
