"""
tests/test_scheduler.py — Unit test untuk core/scheduler.py (jadwal tetap).

APScheduler di-mock agar test tidak blocking.
"""
import unittest
from datetime import datetime
from unittest import mock

import pytz

from core.scheduler import ContentScheduler

_JST = pytz.timezone("Asia/Tokyo")


def _sched(hours=(0, 6, 12, 18)):
    s = ContentScheduler.__new__(ContentScheduler)
    s.posting_hours = list(hours)
    s.daily_target = len(hours)
    s.tz = _JST
    s._scheduler = None
    return s


class TestComputeSchedule(unittest.TestCase):
    def test_returns_one_slot_per_hour(self):
        s = _sched()
        slots = s.compute_schedule(_JST.localize(datetime(2026, 6, 28, 9, 0)))
        self.assertEqual([x.hour for x in slots], [0, 6, 12, 18])

    def test_all_minutes_zero(self):
        s = _sched()
        slots = s.compute_schedule(_JST.localize(datetime(2026, 6, 28, 9, 0)))
        self.assertTrue(all(x.minute == 0 for x in slots))

    def test_sorted_ascending(self):
        s = _sched(hours=(18, 0, 12, 6))
        slots = s.compute_schedule(_JST.localize(datetime(2026, 6, 28, 9, 0)))
        self.assertEqual([x.hour for x in slots], [0, 6, 12, 18])

    def test_invalid_hours_skipped(self):
        s = _sched(hours=(6, 25, -1, 12))
        slots = s.compute_schedule(_JST.localize(datetime(2026, 6, 28, 9, 0)))
        self.assertEqual([x.hour for x in slots], [6, 12])


class TestUpcomingSlots(unittest.TestCase):
    def test_filters_past(self):
        s = _sched()
        now = _JST.localize(datetime(2026, 6, 28, 9, 0))  # setelah 06:00, sebelum 12:00
        slots = s.upcoming_slots(now)
        self.assertEqual([x.hour for x in slots], [12, 18])

    def test_just_after_midnight_has_three(self):
        s = _sched()
        now = _JST.localize(datetime(2026, 6, 28, 0, 1))  # 00:00 sudah lewat
        slots = s.upcoming_slots(now)
        self.assertEqual([x.hour for x in slots], [6, 12, 18])


class TestStart(unittest.TestCase):
    def test_start_adds_cron_job(self):
        s = _sched()
        fake = mock.MagicMock()
        with mock.patch("apscheduler.schedulers.blocking.BlockingScheduler", return_value=fake):
            s.start()
        self.assertTrue(fake.add_job.called)
        # cron dengan jam gabungan
        _, kwargs = fake.add_job.call_args
        self.assertEqual(kwargs.get("trigger"), "cron")
        self.assertEqual(kwargs.get("hour"), "0,6,12,18")
        fake.start.assert_called_once()

    def test_run_cycle_invokes_orchestrator(self):
        s = _sched()
        fake_result = mock.MagicMock(success=True, content_id=1)
        with mock.patch("orchestrator.Orchestrator") as MockOrch:
            MockOrch.return_value.run_cycle.return_value = fake_result
            s._run_cycle()
        MockOrch.return_value.run_cycle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
