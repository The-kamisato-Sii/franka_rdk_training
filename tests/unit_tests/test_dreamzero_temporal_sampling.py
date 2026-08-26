"""CPU-only tests for the explicit DreamZero RoboTwin2 temporal schedule."""

from __future__ import annotations

import unittest

import numpy as np

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    MultiAnchorTemporalConfig,
    require_multi_anchor_temporal_indices,
)


class DreamZeroTemporalSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.language = np.asarray(["same task"] * 128)

    def test_legacy_action48_schedule_is_unchanged(self) -> None:
        cfg = MultiAnchorTemporalConfig(
            max_chunk_size=1,
            macro_stride=48,
            action_horizon=48,
            video_in_chunk_offsets=(0, 6, 12, 18, 24, 30, 36, 42),
        )

        temporal = require_multi_anchor_temporal_indices(
            0, self.language, len(self.language), cfg
        )

        np.testing.assert_array_equal(
            temporal.video, np.arange(0, 49, 6, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            temporal.action, np.arange(48, dtype=np.int64)
        )
        np.testing.assert_array_equal(temporal.state, np.asarray([0]))

    def test_action16_schedule_matches_requested_indices_exactly(self) -> None:
        cfg = MultiAnchorTemporalConfig(
            max_chunk_size=1,
            macro_stride=16,
            action_horizon=16,
            video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
            video_boundary_offset=15,
        )

        temporal = require_multi_anchor_temporal_indices(
            0, self.language, len(self.language), cfg
        )

        np.testing.assert_array_equal(
            temporal.video, np.asarray([0, 1, 3, 5, 7, 9, 11, 13, 15])
        )
        np.testing.assert_array_equal(
            temporal.action, np.arange(16, dtype=np.int64)
        )
        np.testing.assert_array_equal(temporal.state, np.asarray([0]))
        np.testing.assert_array_equal(
            temporal.action[::2], np.asarray([0, 2, 4, 6, 8, 10, 12, 14])
        )

    def test_nonzero_anchor_returns_relative_offsets(self) -> None:
        cfg = MultiAnchorTemporalConfig(
            max_chunk_size=1,
            macro_stride=16,
            action_horizon=16,
            video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
            video_boundary_offset=15,
        )

        temporal = require_multi_anchor_temporal_indices(
            10, self.language, len(self.language), cfg
        )

        np.testing.assert_array_equal(
            temporal.video, np.asarray([0, 1, 3, 5, 7, 9, 11, 13, 15])
        )
        np.testing.assert_array_equal(
            temporal.action, np.arange(16, dtype=np.int64)
        )
        np.testing.assert_array_equal(temporal.state, np.asarray([0]))

    def test_motion_flow_extent_never_selects_zero_padded_tail(self) -> None:
        cfg = MultiAnchorTemporalConfig(
            max_chunk_size=1,
            macro_stride=16,
            action_horizon=16,
            video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
            video_boundary_offset=15,
            required_window_extent=20,
        )

        # Anchor 107 needs frame 127 for motion[14] -> frame[20] and is valid.
        current = require_multi_anchor_temporal_indices(
            107, self.language, len(self.language), cfg
        )
        np.testing.assert_array_equal(
            current.action[::2], np.asarray([0, 2, 4, 6, 8, 10, 12, 14])
        )
        self.assertEqual(107 + int(current.action[::2][-1]) + 6, 127)

        # At dataset index 108 the sampler falls back to the previous complete
        # same-language anchor (92).  Returned values remain relative to 108,
        # and every selected i->i+6 flow is still backed by real episode data.
        fallback = require_multi_anchor_temporal_indices(
            108, self.language, len(self.language), cfg
        )
        np.testing.assert_array_equal(
            fallback.action, np.arange(-16, 0, dtype=np.int64)
        )
        np.testing.assert_array_equal(fallback.state, np.asarray([-16]))
        absolute_motion_rows = 108 + fallback.action[::2]
        np.testing.assert_array_equal(
            absolute_motion_rows,
            np.asarray([92, 94, 96, 98, 100, 102, 104, 106]),
        )
        self.assertLess(int(absolute_motion_rows[-1]) + 6, len(self.language))

    def test_masked_motion_tail_keeps_last_complete_action_video_window(self) -> None:
        cfg = MultiAnchorTemporalConfig(
            max_chunk_size=1,
            macro_stride=16,
            action_horizon=16,
            video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
            video_boundary_offset=15,
            # No required_window_extent=20: padded tail flow is handled by a
            # branch-specific loss mask instead of rejecting RGB/action data.
        )

        # For length 128, anchor 112 is the final full RGB/action window.
        tail = require_multi_anchor_temporal_indices(
            112, self.language, len(self.language), cfg
        )
        np.testing.assert_array_equal(
            tail.video, np.asarray([0, 1, 3, 5, 7, 9, 11, 13, 15])
        )
        np.testing.assert_array_equal(tail.action, np.arange(16, dtype=np.int64))
        absolute_motion_rows = 112 + tail.action[::2]
        np.testing.assert_array_equal(
            absolute_motion_rows,
            np.asarray([112, 114, 116, 118, 120, 122, 124, 126]),
        )
        np.testing.assert_array_equal(
            absolute_motion_rows + 6 < len(self.language),
            np.asarray([True, True, True, True, True, False, False, False]),
        )

        # The next anchor cannot provide 16 real actions/RGB frames and still
        # falls back to the previous complete 16-step window.
        fallback = require_multi_anchor_temporal_indices(
            113, self.language, len(self.language), cfg
        )
        np.testing.assert_array_equal(fallback.action, np.arange(-16, 0))

    def test_nonuniform_schedule_requires_explicit_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            MultiAnchorTemporalConfig(
                max_chunk_size=1,
                macro_stride=16,
                action_horizon=16,
                video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
            )

    def test_nonshared_boundary_rejects_multiple_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "only when max_chunk_size=1"):
            MultiAnchorTemporalConfig(
                max_chunk_size=4,
                macro_stride=16,
                action_horizon=16,
                video_in_chunk_offsets=(0, 1, 3, 5, 7, 9, 11, 13),
                video_boundary_offset=15,
            )


if __name__ == "__main__":
    unittest.main()
