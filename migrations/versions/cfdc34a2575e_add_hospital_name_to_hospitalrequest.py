"""Remove hospital_name from HospitalRequest

Revision ID: cfdc34a2575e
Revises: 
Create Date: 2024-03-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cfdc34a2575e'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Remove hospital_name column from hospital_request table
    op.drop_column('hospital_request', 'hospital_name')


def downgrade():
    # Add hospital_name column back to hospital_request table
    op.add_column('hospital_request', sa.Column('hospital_name', sa.String(255), nullable=False))
