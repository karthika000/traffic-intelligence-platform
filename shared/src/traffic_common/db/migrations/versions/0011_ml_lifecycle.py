"""ML lifecycle: drift check grouping and status, drift mode, retraining recommendations,
promotion decisions, anomaly scores in congestion_history, report style in simulated_reports

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-17 15:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0011'
down_revision: str | None = '0010'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json(name: str, default: str = "'{}'::jsonb") -> sa.Column:
    return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text(default), nullable=False)


def upgrade() -> None:
    op.add_column('drift_reports', sa.Column('check_id', sa.Uuid(), nullable=True))
    op.add_column('drift_reports', sa.Column('status', sa.String(length=20), nullable=True))
    op.create_index('ix_drift_reports_check_id', 'drift_reports', ['check_id'])
    op.create_check_constraint(op.f('ck_drift_reports_status_valid'), 'drift_reports', "status IS NULL OR status IN ('ok', 'warning', 'drifted', 'no_data')")

    op.add_column('model_versions', _json('promotion'))
    op.add_column('congestion_history', sa.Column('anomaly_score', sa.Float(), nullable=True))
    op.add_column('congestion_history', sa.Column('anomaly_threshold', sa.Float(), nullable=True))
    op.add_column('simulated_reports', sa.Column('text_style', sa.String(length=20), nullable=True))

    op.create_table(
        'drift_mode',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        _json('settings'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_by', sa.String(length=50), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('id = 1', name=op.f('ck_drift_mode_single_row')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_drift_mode')),
    )
    op.create_table(
        'drift_mode_events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        _json('settings'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('changed_by', sa.String(length=50), nullable=True),
        sa.Column('changed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_drift_mode_events')),
    )
    op.create_index('ix_drift_mode_events_changed_at', 'drift_mode_events', ['changed_at'])
    op.create_table(
        'retraining_recommendations',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('model_name', sa.String(length=200), nullable=False),
        sa.Column('model_version', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), server_default=sa.text("'recommended'"), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('alert_count', sa.Integer(), server_default=sa.text('1'), nullable=False),
        _json('drifted_features', "'[]'::jsonb"),
        sa.Column('first_alert_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_alert_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_alert_id', sa.Uuid(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        _json('resolution'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("status IN ('recommended', 'retraining', 'resolved')", name=op.f('ck_retraining_recommendations_status_valid')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_retraining_recommendations')),
    )
    op.create_index('uq_retraining_recommendations_open', 'retraining_recommendations', ['model_name'], unique=True, postgresql_where=sa.text("status <> 'resolved'"))
    op.create_index('ix_retraining_recommendations_created_at', 'retraining_recommendations', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_retraining_recommendations_created_at', table_name='retraining_recommendations')
    op.drop_index('uq_retraining_recommendations_open', table_name='retraining_recommendations', postgresql_where=sa.text("status <> 'resolved'"))
    op.drop_table('retraining_recommendations')
    op.drop_index('ix_drift_mode_events_changed_at', table_name='drift_mode_events')
    op.drop_table('drift_mode_events')
    op.drop_table('drift_mode')
    op.drop_column('simulated_reports', 'text_style')
    op.drop_column('congestion_history', 'anomaly_threshold')
    op.drop_column('congestion_history', 'anomaly_score')
    op.drop_column('model_versions', 'promotion')
    op.drop_constraint(op.f('ck_drift_reports_status_valid'), 'drift_reports', type_='check')
    op.drop_index('ix_drift_reports_check_id', table_name='drift_reports')
    op.drop_column('drift_reports', 'status')
    op.drop_column('drift_reports', 'check_id')
