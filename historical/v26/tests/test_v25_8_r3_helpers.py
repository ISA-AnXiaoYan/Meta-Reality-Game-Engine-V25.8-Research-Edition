from __future__ import annotations

import unittest

from global_yolo_polygon_contract import has_usable_polygon, summarize_polygon_contract
from monotonic_rate_deadline import rate_deadline_due


class PolygonContractTests(unittest.TestCase):
    def test_empty_enabled_shard_is_valid(self) -> None:
        report = summarize_polygon_contract(
            persons_total=0,
            persons_with_polygon=0,
            polygon_output_enabled=True,
        )
        self.assertTrue(report["polygon_contract_ok"])
        self.assertEqual(report["polygon_contract_state"], "valid_empty")

    def test_every_detected_person_requires_polygon(self) -> None:
        report = summarize_polygon_contract(
            persons_total=2,
            persons_with_polygon=1,
            polygon_output_enabled=True,
        )
        self.assertFalse(report["polygon_contract_ok"])
        self.assertEqual(report["published_persons_without_polygon"], 1)

    def test_polygon_requires_at_least_three_points(self) -> None:
        self.assertFalse(has_usable_polygon([[[1, 2], [3, 4]]]))
        self.assertTrue(has_usable_polygon([[[1, 2], [3, 4], [5, 6]]]))


class RateDeadlineTests(unittest.TestCase):
    def test_12hz_deadline_is_not_quantized_to_10hz_at_30hz_render(self) -> None:
        deadline = 0.0
        count = 0
        for frame in range(1, 301):
            due, deadline = rate_deadline_due(deadline, frame / 30.0, 12.0)
            count += int(due)
        self.assertGreaterEqual(count, 119)
        self.assertLessEqual(count, 121)


if __name__ == "__main__":
    unittest.main()
