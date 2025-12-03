"""
Routes package - exports all API routers
"""
from app.routes.job_data_ingestion import router as job_data_router

__all__ = ["job_data_router"]
