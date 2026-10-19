"""Drop detection_aggregates: per-class counts per window were never written by any service
(traffic_metrics.counts_by_class holds them)

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-18 22:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0014'
down_revision: str | None = '0013'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index('ix_detection_aggregates_window_start', table_name='detection_aggregates')
    op.drop_index('ix_detection_aggregates_camera_id_window_start', table_name='detection_aggregates')
    op.drop_table('detection_aggregates')


def downgrade() -> None:
    op.create_table('detection_aggregates',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('camera_id', sa.String(length=64), nullable=False),
    sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
    sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
    sa.Column('object_class', sa.String(length=50), nullable=False),
    sa.Column('object_count', sa.Integer(), nullable=False),
    sa.Column('avg_confidence', sa.Double(), nullable=True),
    sa.Column('model_name', sa.String(length=200), nullable=False),
    sa.Column('model_version', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('object_count >= 0', name=op.f('ck_detection_aggregates_object_count_non_negative')),
    sa.CheckConstraint('window_end > window_start', name=op.f('ck_detection_aggregates_window_ordered')),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], name=op.f('fk_detection_aggregates_camera_id_cameras')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_detection_aggregates')),
    sa.UniqueConstraint('camera_id', 'window_start', 'object_class', 'model_name', 'model_version', name='uq_detection_aggregates_window_class_model')
    )
    op.create_index('ix_detection_aggregates_camera_id_window_start', 'detection_aggregates', ['camera_id', 'window_start'], unique=False)
    op.create_index('ix_detection_aggregates_window_start', 'detection_aggregates', ['window_start'], unique=False)
