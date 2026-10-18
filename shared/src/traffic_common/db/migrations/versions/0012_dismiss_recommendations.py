"""retraining_recommendations: a "dismissed" status (closed by hand, with a reason)

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-18 12:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0012'
down_revision: str | None = '0011'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OPEN = "status IN ('recommended', 'retraining')"


def upgrade() -> None:
    op.drop_constraint(op.f('ck_retraining_recommendations_status_valid'), 'retraining_recommendations', type_='check')
    op.create_check_constraint(op.f('ck_retraining_recommendations_status_valid'), 'retraining_recommendations', "status IN ('recommended', 'retraining', 'resolved', 'dismissed')")
    op.drop_index('uq_retraining_recommendations_open', table_name='retraining_recommendations', postgresql_where=sa.text("status <> 'resolved'"))
    op.create_index('uq_retraining_recommendations_open', 'retraining_recommendations', ['model_name'], unique=True, postgresql_where=sa.text(OPEN))


def downgrade() -> None:
    op.execute("UPDATE retraining_recommendations SET status = 'resolved' WHERE status = 'dismissed'")
    op.drop_index('uq_retraining_recommendations_open', table_name='retraining_recommendations', postgresql_where=sa.text(OPEN))
    op.create_index('uq_retraining_recommendations_open', 'retraining_recommendations', ['model_name'], unique=True, postgresql_where=sa.text("status <> 'resolved'"))
    op.drop_constraint(op.f('ck_retraining_recommendations_status_valid'), 'retraining_recommendations', type_='check')
    op.create_check_constraint(op.f('ck_retraining_recommendations_status_valid'), 'retraining_recommendations', "status IN ('recommended', 'retraining', 'resolved')")
