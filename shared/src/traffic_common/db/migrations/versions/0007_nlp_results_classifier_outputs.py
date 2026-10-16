"""nlp_results: incident type, confidences, probabilities, camera match

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-16 15:10:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0007'
down_revision: str | None = '0006'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INCIDENT_TYPES = ('accident', 'breakdown', 'road_works', 'congestion', 'weather_hazard', 'debris', 'police_activity', 'none')


def _json(name: str) -> sa.Column:
    return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False)


def upgrade() -> None:
    op.add_column('nlp_results', sa.Column('incident_type', sa.Enum(*INCIDENT_TYPES, name='incident_type', native_enum=False, create_constraint=True, length=20), nullable=True))
    # Rows written before this revision kept the type in the free-text "category" column.
    op.execute(f"UPDATE nlp_results SET incident_type = CASE WHEN category IN {INCIDENT_TYPES!r} THEN category ELSE 'none' END")
    op.alter_column('nlp_results', 'incident_type', existing_type=sa.String(length=20), nullable=False)
    op.drop_column('nlp_results', 'category')
    op.add_column('nlp_results', sa.Column('camera_confidence', sa.Float(), nullable=True))
    op.add_column('nlp_results', sa.Column('incident_confidence', sa.Float(), nullable=True))
    op.add_column('nlp_results', sa.Column('severity_confidence', sa.Float(), nullable=True))
    op.add_column('nlp_results', _json('probabilities'))
    op.add_column('nlp_results', _json('model_versions'))
    op.add_column('nlp_results', _json('details'))


def downgrade() -> None:
    op.drop_column('nlp_results', 'details')
    op.drop_column('nlp_results', 'model_versions')
    op.drop_column('nlp_results', 'probabilities')
    op.drop_column('nlp_results', 'severity_confidence')
    op.drop_column('nlp_results', 'incident_confidence')
    op.drop_column('nlp_results', 'camera_confidence')
    op.add_column('nlp_results', sa.Column('category', sa.String(length=50), nullable=True))
    op.execute("UPDATE nlp_results SET category = incident_type")
    op.alter_column('nlp_results', 'category', existing_type=sa.String(length=50), nullable=False)
    op.drop_constraint(op.f('ck_nlp_results_incident_type'), 'nlp_results', type_='check')
    op.drop_column('nlp_results', 'incident_type')
