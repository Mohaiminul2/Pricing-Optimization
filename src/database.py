"""
SQLAlchemy ORM models for transaction data
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    """Transaction table"""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    invoice_no = Column(String, unique=True, index=True)
    invoice_date = Column(DateTime, index=True)
    quantity = Column(Integer)
    unit_price = Column(Float)
    product_id = Column(String, index=True)
    product_name = Column(String)
    category = Column(String, index=True)
    customer_id = Column(String, index=True)
    country = Column(String)
    revenue = Column(Float)  # quantity * unit_price
    promotion_flag = Column(Boolean, default=False)  # True if price dropped >20% vs prior tx
    is_winter = Column(Integer, default=0)  # Month in [Nov, Dec, Jan, Feb]
    is_holiday_season = Column(Integer, default=0)  # Month in [Nov, Dec]
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_path):
    """Create SQLAlchemy engine"""
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Create session for database operations"""
    Session = sessionmaker(bind=engine)
    return Session()
