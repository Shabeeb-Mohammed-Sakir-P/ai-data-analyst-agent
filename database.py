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
    status = Column(String, default="uploaded")  # uploaded -> processing -> complete/failed
    results_json = Column(Text, nullable=True)  # stores the pipeline results as JSON text
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    error_message = Column(Text, nullable=True)
    owner_id = Column(String, nullable=True)  # links to User.id; nullable for now, for backward compatibility


class User(Base):
    """
    Represents a registered user of the app.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Creates the actual database file and tables, if they don't already exist
Base.metadata.create_all(bind=engine)


def get_db():
    """
    Provides a database session for a single request, and guarantees
    it's properly closed afterward — even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()