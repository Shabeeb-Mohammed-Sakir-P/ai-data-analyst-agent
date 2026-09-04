from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Dataset(Base):
    """
    Represents one uploaded dataset and its analysis lifecycle.
    """
    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True)
    filename = Column(String)
    status = Column(String, default="uploaded")
    current_step = Column(String, nullable=True)
    results_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    error_message = Column(Text, nullable=True)
    owner_id = Column(String, nullable=True)


class User(Base):
    """
    Represents a registered user of the app.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()