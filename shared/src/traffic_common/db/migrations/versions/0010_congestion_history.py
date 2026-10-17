"""congestion_history: every congestion classification, for analytics

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-17 09:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0010'
down_revision: str | None = '0009'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'congestion_history',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('camera_id', sa.String(length=64), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('level', sa.Enum('free_flow', 'moderate', 'heavy', 'standstill', name='congestion_level', native_enum=False, create_constraint=True, length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], name=op.f('fk_congestion_history_camera_id_cameras')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_congestion_history')),
    )
    op.create_index('ix_congestion_history_camera_id_window_end', 'congestion_history', ['camera_id', 'window_end'])
    op.create_index('ix_congestion_history_window_end', 'congestion_history', ['window_end'])


def downgrade() -> None:
    op.drop_index('ix_congestion_history_window_end', table_name='congestion_history')
    op.drop_index('ix_congestion_history_camera_id_window_end', table_name='congestion_history')
    op.drop_table('congestion_history')
