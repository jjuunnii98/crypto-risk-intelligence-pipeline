from src.data.database import engine
from src.data.db_models import Base

def init_db() -> None:
    """
    Initialize database and create all tables.

    This will create:
    - market_snapshots
    - news_articles
    - risk_snapshots
    - candle_snapshots  ← NEW
    """
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()