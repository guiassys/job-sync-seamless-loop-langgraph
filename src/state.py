# src/state.py

from typing import TypedDict


class VideoState(TypedDict):
    """Contrato de estado compartilhado entre os nós do pipeline Job Sync Seamless Loop."""

    user_prompt: str
    visual_specification: str
    loop_validation_metrics: str
    video_filepath: str
    final_report: str