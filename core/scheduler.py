"""
core/scheduler.py — JST-aware content scheduler.

Menjadwalkan `daily_target` siklus per hari, tersebar merata di window
aktif (default 07:00–24:00 JST) dengan jitter acak ±jitter_minutes agar
posting terlihat alami, bukan robotik.

Cara kerja:
  1. compute_schedule() hitung N slot datetime JST untuk satu hari
  2. upcoming_slots() saring slot yang belum lewat (mulai dari sekarang)
  3. start() tambahkan semua slot ke APScheduler, tambahkan cron 00:01 JST
     untuk reschedule otomatis setiap hari
"""
import random
from datetime import datetime, date, time, timedelta
from typing import List, Optional

import pytz

from core.config import load_config
from core.logger import get_logger

logger = get_logger("scheduler")


class ContentScheduler:
    """Scheduler harian berbasis APScheduler dengan slot JST + jitter."""

    def __init__(self):
        cfg = load_config()
        sched = cfg.get("scheduling", {})
        hours = sched.get("active_hours_jst", {})

        self.daily_target:  int = sched.get("daily_target", 20)
        self.start_hour:    int = hours.get("start", 7)
        self.end_hour:      int = hours.get("end", 24)
        self.jitter_minutes: int = sched.get("jitter_minutes", 10)
        self.tz = pytz.timezone("Asia/Tokyo")
        self._scheduler = None

    # ------------------------------------------------------------------
    # Kalkulasi jadwal
    # ------------------------------------------------------------------

    def compute_schedule(self, reference_dt: Optional[datetime] = None) -> List[datetime]:
        """
        Hitung `daily_target` slot waktu JST untuk tanggal reference_dt (default hari ini).

        Algoritma:
          • Window aktif: [start_hour, end_hour) dalam menit dari midnight
            (end_hour=24 diperlakukan sebagai 23:59 karena time() maks 23:59)
          • Interval = window / daily_target menit
          • Slot ke-i: titik tengah bucket i + jitter uniform(–N, +N) menit
          • Diclamped ke dalam window, diurutkan ASC
        """
        if reference_dt is None:
            reference_dt = datetime.now(self.tz)

        today: date = reference_dt.date()

        # Menit dari midnight (0–1439)
        start_min = self.start_hour * 60
        end_min   = min(self.end_hour * 60, 24 * 60 - 1)  # maks 23:59 = 1439
        window    = end_min - start_min  # total window dalam menit

        if window <= 0 or self.daily_target <= 0:
            return []

        interval = window / self.daily_target

        slots: List[datetime] = []
        for i in range(self.daily_target):
            center = start_min + (i + 0.5) * interval
            jitter = random.uniform(-self.jitter_minutes, self.jitter_minutes)
            actual = center + jitter

            # Clamp ke dalam window
            actual = max(float(start_min), min(float(end_min), actual))

            total_minutes = int(actual)
            hour   = total_minutes // 60
            minute = total_minutes % 60

            slot_naive = datetime.combine(today, time(hour=hour, minute=minute))
            slot_aware = self.tz.localize(slot_naive)
            slots.append(slot_aware)

        return sorted(slots)

    def upcoming_slots(self, reference_dt: Optional[datetime] = None) -> List[datetime]:
        """Kembalikan slot hari ini yang belum lewat."""
        now = reference_dt or datetime.now(self.tz)
        return [s for s in self.compute_schedule(now) if s > now]

    # ------------------------------------------------------------------
    # APScheduler
    # ------------------------------------------------------------------

    def start(self):
        """
        Mulai scheduler (blocking).
        Hitung jadwal hari ini, jadwalkan semua slot, dan tambahkan
        cron 00:01 JST untuk reschedule otomatis setiap tengah malam.
        """
        from apscheduler.schedulers.blocking import BlockingScheduler

        self._scheduler = BlockingScheduler(timezone=self.tz)

        # Reschedule otomatis setiap 00:01 JST
        self._scheduler.add_job(
            self._reschedule_daily,
            trigger="cron",
            hour=0, minute=1,
            id="daily_reschedule",
            replace_existing=True,
        )

        # Jadwalkan hari ini
        self._schedule_today()

        logger.info(
            "ContentScheduler dimulai: %d/hari | JST %02d:00–%02d:00 | jitter ±%d mnt",
            self.daily_target, self.start_hour, self.end_hour, self.jitter_minutes,
        )
        self._scheduler.start()

    def _schedule_today(self):
        """Tambahkan DateTrigger job untuk setiap slot hari ini yang belum lewat."""
        slots = self.upcoming_slots()
        for i, slot in enumerate(slots):
            job_id = f"content_{slot.strftime('%Y%m%d_%H%M')}_{i}"
            self._scheduler.add_job(
                self._run_cycle,
                trigger="date",
                run_date=slot,
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300,  # 5 menit grace period
            )
        logger.info("Dijadwalkan %d/%d slot hari ini", len(slots), self.daily_target)

    def _reschedule_daily(self):
        """Callback 00:01 JST — hapus job lama dan jadwalkan hari baru."""
        if self._scheduler is None:
            return
        removed = 0
        for job in self._scheduler.get_jobs():
            if job.id.startswith("content_"):
                job.remove()
                removed += 1
        logger.info("Reschedule harian: %d job lama dihapus", removed)
        self._schedule_today()

    def _run_cycle(self):
        """Jalankan satu siklus Orchestrator (dipanggil oleh APScheduler)."""
        logger.info(">>> Mulai siklus terjadwal (%s JST)",
                    datetime.now(self.tz).strftime("%H:%M"))
        try:
            from orchestrator import Orchestrator
            result = Orchestrator().run_cycle()
            if result.success:
                logger.info("Siklus selesai: konten #%s berhasil", result.content_id)
            else:
                logger.warning("Siklus selesai tanpa konten: %s", result.reason or result.error)
        except Exception as e:
            logger.error("Error dalam siklus terjadwal: %s", e, exc_info=True)
