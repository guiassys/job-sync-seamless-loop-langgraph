# src/flow_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.state import VideoState

class VideoGenerationStrategy(ABC):
    """Abstract base class for video generation strategies."""

    @abstractmethod
    def execute(self, state: VideoState) -> Dict[str, Any]:
        """Execute the video generation workflow."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this strategy."""
        pass

class OriginalVideoGenerationStrategy(VideoGenerationStrategy):
    """Original strategy implementation - maintains full pipeline behavior with loop validation."""

    def __init__(self):
        self.director_node = None
        self.video_engineer_node = None
        self.validator_node = None
        self.editor_node = None

    def execute(self, state: VideoState) -> Dict[str, Any]:
        """Execute the full multi-node seamless loop workflow."""
        # Import here to avoid circular imports during initialization
        from src.nodes import director_node, video_engineer_node, validator_node, editor_node

        print("\n🎬 Executing Original Seamless Loop Video Generation Flow")

        # Execute nodes in sequence
        director_result = director_node(state)
        state.update(director_result)

        video_engineer_result = video_engineer_node(state)
        state.update(video_engineer_result)

        validator_result = validator_node(state)
        state.update(validator_result)

        editor_result = editor_node(state)

        return {
            "final_report": editor_result["final_report"],
            "video_filepath": state["video_filepath"],
            "visual_specification": state["visual_specification"],
            "loop_validation_metrics": state["loop_validation_metrics"]
        }

    def get_name(self) -> str:
        return "original"

class FastVideoGenerationStrategy(VideoGenerationStrategy):
    """Fast generation strategy - simplified workflow bypassing intermediate validation steps."""

    def __init__(self):
        self.director_node = None
        self.video_engineer_node = None
        self.editor_node = None

    def execute(self, state: VideoState) -> Dict[str, Any]:
        """Execute a faster, simplified video generation workflow."""
        # Import here to avoid circular imports during initialization
        from src.nodes import director_node, video_engineer_node, editor_node

        print("\n⚡ Executing Fast Seamless Loop Video Generation Flow")

        # Execute director and video engineer nodes directly
        director_result = director_node(state)
        state.update(director_result)

        video_engineer_result = video_engineer_node(state)
        state.update(video_engineer_result)

        state["loop_validation_metrics"] = (
            "Métricas detalhadas de loop omitidas no modo rápido. "
            "O arquivo passou pela verificação de decodificação e conteúdo na exportação."
        )

        # Execute editor node
        editor_result = editor_node(state)

        return {
            "final_report": editor_result["final_report"],
            "video_filepath": state["video_filepath"],
            "visual_specification": state["visual_specification"],
            "loop_validation_metrics": state["loop_validation_metrics"]
        }

    def get_name(self) -> str:
        return "fast"