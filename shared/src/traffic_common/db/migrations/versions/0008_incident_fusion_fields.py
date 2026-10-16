"""incidents: type, version, confidence breakdown, lifecycle times; report stage

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-16 16:00:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0008'
down_revision: str | None = '0007'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INCIDENT_TYPES = ('accident', 'breakdown', 'road_works', 'congestion', 'weather_hazard', 'debris', 'police_activity', 'none')


def _json(name: str) -> sa.Column:
    return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False)


def upgrade() -> None:
    op.add_column('incidents', sa.Column('incident_type', sa.Enum(*INCIDENT_TYPES, name='incident_type', native_enum=False, create_constraint=True, length=20), server_default='none', nullable=False))
    op.add_column('incidents', sa.Column('version', sa.Integer(), server_default=sa.text('1'), nullable=False))
    op.add_column('incidents', _json('confidence_components'))
    op.add_column('incidents', sa.Column('vision_degraded', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.add_column('incidents', sa.Column('suspected_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('incidents', sa.Column('confirmed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('incidents', _json('details'))
    op.add_column('incident_evidence', _json('details'))
    op.add_column('nlp_results', sa.Column('stage', sa.String(length=20), server_default=sa.text("'active'"), nullable=False))


def downgrade() -> None:
    op.drop_column('nlp_results', 'stage')
    op.drop_column('incident_evidence', 'details')
    op.drop_column('incidents', 'details')
    op.drop_column('incidents', 'confirmed_at')
    op.drop_column('incidents', 'suspected_at')
    op.drop_column('incidents', 'vision_degraded')
    op.drop_column('incidents', 'confidence_components')
    op.drop_column('incidents', 'version')
    op.drop_constraint(op.f('ck_incidents_incident_type'), 'incidents', type_='check')
    op.drop_column('incidents', 'incident_type')
