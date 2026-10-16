"""summaries: headline, incident version, generation path, prompt, latency, faithfulness

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-16 20:10:00.000000
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0009'
down_revision: str | None = '0008'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json(name: str) -> sa.Column:
    return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False)


def upgrade() -> None:
    op.add_column('summaries', sa.Column('incident_version', sa.Integer(), nullable=True))
    op.add_column('summaries', sa.Column('headline', sa.String(), nullable=True))
    op.add_column('summaries', sa.Column('generation', sa.String(length=20), nullable=True))
    op.add_column('summaries', sa.Column('prompt_version', sa.String(length=50), nullable=True))
    op.add_column('summaries', sa.Column('latency_ms', sa.Float(), nullable=True))
    op.add_column('summaries', _json('faithfulness'))
    op.add_column('summaries', _json('context'))
    op.create_check_constraint(op.f('ck_summaries_generation_valid'), 'summaries', "generation IS NULL OR generation IN ('model', 'template')")


def downgrade() -> None:
    op.drop_constraint(op.f('ck_summaries_generation_valid'), 'summaries', type_='check')
    op.drop_column('summaries', 'context')
    op.drop_column('summaries', 'faithfulness')
    op.drop_column('summaries', 'latency_ms')
    op.drop_column('summaries', 'prompt_version')
    op.drop_column('summaries', 'generation')
    op.drop_column('summaries', 'headline')
    op.drop_column('summaries', 'incident_version')
