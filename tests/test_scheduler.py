"""
tests/test_scheduler.py — Unit test untuk core/scheduler.py

APScheduler di-mock agar test tidak blocking dan tidak butuh event loop.
"""
import unittest
from datetime import datetime, time, timedelta
from unittest import mock

import pytz

JST = pytz.timezone("Asia/Tokyo")

_CFG = {
    "scheduling": {
        "daily_target": 20,
        "active_hours_jst": {"start": 7, "end": 24},
        "jitter_minutes": 10,
    }
}

_CFG_SMALL = {
    "scheduling": {
        "daily_target": 5,
        "active_hours_jst": {"start": 9, "end": 18},
        "jitter_minutes": 5,
    }
}


def _make_scheduler(cfg=None):
    from core.scheduler import ContentScheduler
    with mock.patch("core.scheduler.load_config", return_value=cfg or _CFG):
        s = ContentScheduler()
    return s


def _jst(year=2024, month=6, day=1, hour=8, minute=0):
    return JST.localize(datetime(year, month, day, hour, minute))


class TestComputeSchedule(unittest.TestCase):

    def test_returns_correct_count(self):
        s = _make_scheduler()
        ref = _jst(hour=0, minute=0)  # midnight — semua slot belum lewat
        slots = s.compute_schedule(ref)
        self.assertEqual(len(slots), 20)

    def test_slots_within_window(self):
        s = _make_scheduler()
        ref = _jst(hour=0)
        slots = s.compute_schedule(ref)
        for slot in slots:
            self.assertGreaterEqual(slot.hour, 7, f"Slot {slot} sebelum 07:00")
            # 23:59 adalah batas maksimal (end_hour=24 → 23:59)
            self.assertLessEqual(slot.hour * 60 + slot.minute, 23 * 60 + 59)

    def test_slots_sorted_ascending(self):
        s = _make_scheduler()
        ref = _jst(hour=0)
        slots = s.compute_schedule(ref)
        self.assertEqual(slots, sorted(slots))

    def test_all_slots_on_same_date(self):
        s = _make_scheduler()
        ref = _jst(year=2024, month=6, day=15, hour=0)
        slots = s.compute_schedule(ref)
        for slot in slots:
            self.assertEqual(slot.date(), ref.date())

    def test_slots_are_jst_aware(self):
        s = _make_scheduler()
        ref = _jst(hour=0)
        slots = s.compute_schedule(ref)
        for slot in slots:
            self.assertIsNotNone(slot.tzinfo)
            # Verifikasi timezone adalah JST (+09:00)
            offset = slot.utcoffset().total_seconds()
            self.assertEqual(offset, 9 * 3600)

    def test_jitter_causes_variation(self):
        """Setiap panggil compute_schedule harus menghasilkan slot yang sedikit berbeda."""
        s = _make_scheduler()
        ref = _jst(hour=0)
        schedule1 = [sl.strftime("%H:%M") for sl in s.compute_schedule(ref)]
        schedule2 = [sl.strftime("%H:%M") for sl in s.compute_schedule(ref)]
        # Dengan jitter ±10 menit, kemungkinan besar ada perbedaan
        # (probabilitas semua 20 sama persis sangat kecil)
        # Jalankan beberapa kali untuk mengurangi flakiness
        any_diff = False
        for _ in range(5):
            s1 = [sl.strftime("%H:%M") for sl in s.compute_schedule(ref)]
            s2 = [sl.strftime("%H:%M") for sl in s.compute_schedule(ref)]
            if s1 != s2:
                any_diff = True
                break
        self.assertTrue(any_diff, "Jitter tidak menghasilkan variasi")

    def test_custom_window_respected(self):
        s = _make_scheduler(_CFG_SMALL)
        ref = _jst(hour=0)
        slots = s.compute_schedule(ref)
        self.assertEqual(len(slots), 5)
        for slot in slots:
            self.assertGreaterEqual(slot.hour, 9)
            self.assertLessEqual(slot.hour, 18)

    def test_zero_target_returns_empty(self):
        cfg = {"scheduling": {"daily_target": 0, "active_hours_jst": {"start": 7, "end": 24}, "jitter_minutes": 0}}
        s = _make_scheduler(cfg)
        ref = _jst(hour=0)
        self.assertEqual(s.compute_schedule(ref), [])

    def test_window_minutes_calculation(self):
        """Window 07:00–24:00 = 1020 menit, 20 slot → interval ~51 menit."""
        s = _make_scheduler()
        ref = _jst(hour=0)
        with mock.patch("random.uniform", return_value=0.0):  # zero jitter
            slots = s.compute_schedule(ref)
        # Dengan nol jitter, verifikasi slot pertama dekat 07:00
        first = slots[0]
        self.assertEqual(first.hour, 7)

    def test_no_jitter_slots_evenly_distributed(self):
        """Tanpa jitter, slot harus terdistribusi merata."""
        s = _make_scheduler(_CFG_SMALL)
        ref = _jst(hour=0)
        with mock.patch("random.uniform", return_value=0.0):
            slots = s.compute_schedule(ref)
        # 5 slot di 9:00-18:00 (540 menit window), interval = 108 menit
        # Centers: 9:54, 11:42, 13:30, 15:18, 17:06
        self.assertEqual(len(slots), 5)
        # Interval antar slot harus kira-kira sama
        gaps = [(slots[i+1] - slots[i]).seconds // 60 for i in range(len(slots)-1)]
        for gap in gaps:
            self.assertAlmostEqual(gap, gaps[0], delta=2)


class TestUpcomingSlots(unittest.TestCase):

    def test_filters_past_slots(self):
        s = _make_scheduler()
        # Reference jam 18:00 — banyak slot sudah lewat
        ref = _jst(hour=18, minute=0)
        upcoming = s.upcoming_slots(ref)
        # Semua slot harus setelah 18:00
        for slot in upcoming:
            self.assertGreater(slot, ref)

    def test_all_slots_upcoming_at_midnight(self):
        s = _make_scheduler()
        # Jam 00:00 — semua slot belum lewat
        ref = _jst(hour=0, minute=0)
        upcoming = s.upcoming_slots(ref)
        self.assertEqual(len(upcoming), 20)

    def test_no_upcoming_after_end_hour(self):
        s = _make_scheduler()
        # Jam 23:59 — semua slot (sampai 23:59) sudah lewat atau lewat
        ref = _jst(hour=23, minute=59)
        upcoming = s.upcoming_slots(ref)
        self.assertLessEqual(len(upcoming), 1)  # mungkin ada 1 slot di 23:59 tepat

    def test_upcoming_count_decreases_through_day(self):
        s = _make_scheduler()
        morning = _jst(hour=8)
        afternoon = _jst(hour=14)
        evening = _jst(hour=20)
        n_morning = len(s.upcoming_slots(morning))
        n_afternoon = len(s.upcoming_slots(afternoon))
        n_evening = len(s.upcoming_slots(evening))
        self.assertGreaterEqual(n_morning, n_afternoon)
        self.assertGreaterEqual(n_afternoon, n_evening)


class TestRunCycle(unittest.TestCase):

    def test_run_cycle_calls_orchestrator(self):
        s = _make_scheduler()
        mock_result = mock.MagicMock()
        mock_result.success = True
        mock_result.content_id = 42

        # Orchestrator diimport secara lazy di dalam _run_cycle, patch di sumbernya
        with mock.patch("orchestrator.Orchestrator") as MockOrch:
            MockOrch.return_value.run_cycle.return_value = mock_result
            s._run_cycle()

        MockOrch.return_value.run_cycle.assert_called_once()

    def test_run_cycle_handles_exception(self):
        """Exception dalam orchestrator tidak boleh propagate."""
        s = _make_scheduler()

        with mock.patch("orchestrator.Orchestrator", side_effect=Exception("crash")):
            s._run_cycle()  # harus tidak raise


class TestSchedulerStart(unittest.TestCase):

    def test_start_adds_daily_reschedule_job(self):
        s = _make_scheduler()
        mock_scheduler = mock.MagicMock()

        # BlockingScheduler diimport lazy di start(), patch di sumbernya
        with mock.patch("apscheduler.schedulers.blocking.BlockingScheduler",
                        return_value=mock_scheduler), \
             mock.patch.object(s, "_schedule_today"):
            mock_scheduler.start.side_effect = KeyboardInterrupt
            try:
                s.start()
            except KeyboardInterrupt:
                pass

        # daily_reschedule job harus ditambahkan
        calls = [str(c) for c in mock_scheduler.add_job.call_args_list]
        self.assertTrue(any("daily_reschedule" in c for c in calls),
                        "daily_reschedule job tidak ditemukan")

    def test_schedule_today_adds_date_jobs(self):
        s = _make_scheduler()
        mock_scheduler = mock.MagicMock()
        s._scheduler = mock_scheduler

        ref = _jst(hour=7, minute=0)  # awal window — semua 20 slot valid
        with mock.patch.object(s, "upcoming_slots", return_value=[
            _jst(hour=8), _jst(hour=10), _jst(hour=12),
        ]):
            s._schedule_today()

        self.assertEqual(mock_scheduler.add_job.call_count, 3)

    def test_reschedule_daily_removes_old_jobs(self):
        s = _make_scheduler()
        mock_scheduler = mock.MagicMock()
        s._scheduler = mock_scheduler

        old_job1 = mock.MagicMock()
        old_job1.id = "content_20240601_0830_0"
        old_job2 = mock.MagicMock()
        old_job2.id = "content_20240601_1000_1"
        daily_job = mock.MagicMock()
        daily_job.id = "daily_reschedule"

        mock_scheduler.get_jobs.return_value = [old_job1, old_job2, daily_job]

        with mock.patch.object(s, "_schedule_today"):
            s._reschedule_daily()

        old_job1.remove.assert_called_once()
        old_job2.remove.assert_called_once()
        daily_job.remove.assert_not_called()

    def test_reschedule_daily_calls_schedule_today(self):
        s = _make_scheduler()
        mock_scheduler = mock.MagicMock()
        mock_scheduler.get_jobs.return_value = []
        s._scheduler = mock_scheduler

        with mock.patch.object(s, "_schedule_today") as mock_st:
            s._reschedule_daily()

        mock_st.assert_called_once()


class TestMainIntegration(unittest.TestCase):
    """Test bahwa main.py --schedule memanggil ContentScheduler.start()."""

    def test_schedule_flag_starts_scheduler(self):
        import sys
        # setup_logger dan init_db diimport lazy di dalam main(), patch di sumbernya
        with mock.patch.object(sys, "argv", ["main.py", "--schedule"]), \
             mock.patch("main._check_env", return_value=True), \
             mock.patch("core.logger.setup_logger", return_value=mock.MagicMock()), \
             mock.patch("database.db_manager.init_db"), \
             mock.patch("core.scheduler.ContentScheduler") as MockSched:
            MockSched.return_value.start.return_value = None
            import main
            main.main()

        MockSched.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
