"""system-scoped anomalies (pipeline-wide degradation)

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-16 14:30:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0006'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('anomalies', sa.Column('scope', sa.String(length=20), server_default=sa.text("'camera'"), nullable=False))
    op.add_column('anomalies', sa.Column('affected_cameras', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False))
    op.alter_column('anomalies', 'camera_id', existing_type=sa.String(length=64), nullable=True)
    op.create_check_constraint(op.f('ck_anomalies_scope_valid'), 'anomalies', "scope IN ('camera', 'system')")
    op.create_check_constraint(op.f('ck_anomalies_camera_for_camera_scope'), 'anomalies', "scope = 'system' OR camera_id IS NOT NULL")


def downgrade() -> None:
    op.execute("DELETE FROM anomalies WHERE scope = 'system'")
    op.drop_constraint(op.f('ck_anomalies_camera_for_camera_scope'), 'anomalies', type_='check')
    op.drop_constraint(op.f('ck_anomalies_scope_valid'), 'anomalies', type_='check')
    op.alter_column('anomalies', 'camera_id', existing_type=sa.String(length=64), nullable=False)
    op.drop_column('anomalies', 'affected_cameras')
    op.drop_column('anomalies', 'scope')
