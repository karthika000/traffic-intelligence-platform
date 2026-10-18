"""incidents.first_evidence_at: the earliest attached evidence, kept apart from start_time

start_time is now the first evidence supporting the incident's type; it used to be the earliest
evidence of any kind, so existing rows get that value as their first_evidence_at.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-18 18:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0013'
down_revision: str | None = '0012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('incidents', sa.Column('first_evidence_at', sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE incidents SET first_evidence_at = start_time")


def downgrade() -> None:
    op.drop_column('incidents', 'first_evidence_at')
