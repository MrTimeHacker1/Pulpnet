"""SQLAlchemy 2.0 engine/session management (sync).

Single shared engine built from `settings.db_url`. `init_db()` creates tables.
Use `session_scope()` for a transactional block or `get_session()` for a session.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        url = get_settings().db_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args)
        _SessionFactory = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
        logger.info("DB engine created for %s", url.split("://", 1)[0])
    return _engine


def init_db() -> None:
    """Create all tables idempotently."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    get_engine()
    assert _SessionFactory is not None
    return _SessionFactory()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
