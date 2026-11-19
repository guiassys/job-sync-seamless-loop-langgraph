import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.tools import (
    _build_ping_pong_frames,
    _stabilize_camera,
    inspect_video,
)


class VideoProcessingTests(unittest.TestCase):
    def _write_video(self, directory: str, frames: list[np.ndarray]) -> str:
        filepath = str(Path(directory) / "test.mp4")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            filepath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (width, height),
        )
        self.assertTrue(writer.isOpened())
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return filepath

    def test_ping_pong_frames_reverse_without_repeating_endpoints(self):
        frames = [Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8)) for index in range(5)]

        result = _build_ping_pong_frames(frames)

        self.assertEqual(len(result), 2 * len(frames) - 2)
        self.assertEqual([int(np.asarray(frame).mean()) for frame in result], [0, 1, 2, 3, 4, 3, 2, 1])

    def test_camera_stabilization_compensates_global_translation(self):
        rng = np.random.default_rng(42)
        base = rng.integers(0, 256, (128, 192, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        shift = np.float32([[1, 0, 6], [0, 1, -4]])
        moved = cv2.warpAffine(
            base,
            shift,
            (base.shape[1], base.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        result = _stabilize_camera(
            [Image.fromarray(base), Image.fromarray(moved)]
        )

        stabilized = np.asarray(result[1])
        center = np.s_[20:-20, 20:-20]
        before_error = np.abs(moved[center].astype(np.int16) - base[center]).mean()
        after_error = np.abs(stabilized[center].astype(np.int16) - base[center]).mean()
        self.assertLess(after_error, before_error * 0.4)

    def test_inspect_video_returns_decoded_video_metrics(self):
        gradient = np.tile(np.arange(64, dtype=np.uint8), (36, 1))
        frames = [
            cv2.merge(
                [
                    np.clip(gradient + index * 5, 0, 255).astype(np.uint8),
                    np.flip(gradient, axis=1),
                    np.full((36, 64), 100, dtype=np.uint8),
                ]
            )
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            metrics = inspect_video(self._write_video(directory, frames))

        self.assertEqual(metrics["frame_count"], 8)
        self.assertEqual((metrics["width"], metrics["height"]), (64, 36))
        self.assertEqual(metrics["fps"], 5.0)
        self.assertGreater(metrics["mean_contrast"], 8)

    def test_inspect_video_rejects_near_black_output(self):
        frames = [np.zeros((36, 64, 3), dtype=np.uint8) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            filepath = self._write_video(directory, frames)
            with self.assertRaisesRegex(RuntimeError, "quase preto"):
                inspect_video(filepath)


if __name__ == "__main__":
    unittest.main()
