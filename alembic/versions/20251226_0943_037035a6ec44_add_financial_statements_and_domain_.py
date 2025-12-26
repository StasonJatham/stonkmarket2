"""add_financial_statements_and_domain_metrics

Revision ID: 037035a6ec44
Revises: c14161221ecb
Create Date: 2025-12-26 09:43:29.344489+00:00

Adds financial statement storage and domain-specific metrics to stock_fundamentals:
- Domain classification (bank, reit, insurer, etc.)
- JSONB columns for quarterly/annual income stmt, balance sheet, cash flow
- Pre-calculated domain metrics: NIM (banks), FFO (REITs), loss ratio (insurers)

This allows us to store financial data locally and minimize yfinance API calls,
since financial statements only change quarterly.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "037035a6ec44"
down_revision: Union[str, None] = "c14161221ecb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add financial statement columns to stock_fundamentals."""
    # Domain classification
    op.add_column(
        "stock_fundamentals",
        sa.Column("domain", sa.String(20), nullable=True, comment="Domain: bank, reit, insurer, utility, biotech, etf, stock"),
    )
    
    # Last earnings date (for detecting when to refresh)
    op.add_column(
        "stock_fundamentals",
        sa.Column("earnings_date", sa.DateTime(timezone=True), nullable=True, comment="Last earnings date"),
    )
    
    # Financial Statements as JSONB (quarterly and annual)
    op.add_column(
        "stock_fundamentals",
        sa.Column("income_stmt_quarterly", JSONB, nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("income_stmt_annual", JSONB, nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("balance_sheet_quarterly", JSONB, nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("balance_sheet_annual", JSONB, nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("cash_flow_quarterly", JSONB, nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("cash_flow_annual", JSONB, nullable=True),
    )
    
    # Domain-Specific Metrics - Banks
    op.add_column(
        "stock_fundamentals",
        sa.Column("net_interest_income", sa.BigInteger, nullable=True, comment="NII from income stmt"),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("net_interest_margin", sa.Numeric(10, 6), nullable=True, comment="NIM = NII / Interest-earning assets"),
    )
    
    # Domain-Specific Metrics - REITs
    op.add_column(
        "stock_fundamentals",
        sa.Column("ffo", sa.BigInteger, nullable=True, comment="Funds From Operations = Net Income + D&A"),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("ffo_per_share", sa.Numeric(12, 4), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("p_ffo", sa.Numeric(12, 4), nullable=True, comment="Price / FFO per share"),
    )
    
    # Domain-Specific Metrics - Insurers
    op.add_column(
        "stock_fundamentals",
        sa.Column("loss_ratio", sa.Numeric(10, 6), nullable=True, comment="Loss adj expense / Premiums"),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("combined_ratio", sa.Numeric(10, 6), nullable=True, comment="Loss ratio + expense ratio"),
    )
    
    # Timestamp for when financial statements were fetched
    op.add_column(
        "stock_fundamentals",
        sa.Column("financials_fetched_at", sa.DateTime(timezone=True), nullable=True),
    )
    
    # Indexes for efficient querying
    op.create_index("idx_stock_fundamentals_domain", "stock_fundamentals", ["domain"])
    op.create_index("idx_stock_fundamentals_next_earnings", "stock_fundamentals", ["next_earnings_date"])


def downgrade() -> None:
    """Remove financial statement columns from stock_fundamentals."""
    # Drop indexes
    op.drop_index("idx_stock_fundamentals_next_earnings", table_name="stock_fundamentals")
    op.drop_index("idx_stock_fundamentals_domain", table_name="stock_fundamentals")
    
    # Drop columns in reverse order
    op.drop_column("stock_fundamentals", "financials_fetched_at")
    op.drop_column("stock_fundamentals", "combined_ratio")
    op.drop_column("stock_fundamentals", "loss_ratio")
    op.drop_column("stock_fundamentals", "p_ffo")
    op.drop_column("stock_fundamentals", "ffo_per_share")
    op.drop_column("stock_fundamentals", "ffo")
    op.drop_column("stock_fundamentals", "net_interest_margin")
    op.drop_column("stock_fundamentals", "net_interest_income")
    op.drop_column("stock_fundamentals", "cash_flow_annual")
    op.drop_column("stock_fundamentals", "cash_flow_quarterly")
    op.drop_column("stock_fundamentals", "balance_sheet_annual")
    op.drop_column("stock_fundamentals", "balance_sheet_quarterly")
    op.drop_column("stock_fundamentals", "income_stmt_annual")
    op.drop_column("stock_fundamentals", "income_stmt_quarterly")
    op.drop_column("stock_fundamentals", "earnings_date")
    op.drop_column("stock_fundamentals", "domain")
