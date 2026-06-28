"""
core/scheduler.py — JST-aware content scheduler (jadwal tetap).

Menjalankan siklus produksi pada jam-jam tetap JST (default 00:00, 06:00,
12:00, 18:00 — setiap 6 jam, 4 tweet/hari). Tidak ada jitter: jadwal pasti.
Setara cron "0 */6 * * *" pada zona waktu Asia/Tokyo.
"""
from datetime import datetime, date, time
from typing import List, Optional

import pytz

from core.config import load_config
from core.logger import get_logger

logger = get_logger("scheduler")


class ContentScheduler:
    """Scheduler harian berbasis APScheduler dengan jam posting tetap (JST)."""

    def __init__(self):
        cfg = load_config()
        sched = cfg.get("scheduling", {})
        self.posting_hours: List[int] = sched.get("posting_hours_jst", [0, 6, 12, 18])
        self.daily_target: int = sched.get("daily_target", len(self.posting_hours))
        self.tz = pytz.timezone("Asia/Tokyo")
        self._scheduler = None

    # ------------------------------------------------------------------
    # Kalkulasi jadwal
    # ------------------------------------------------------------------

    def compute_schedule(self, reference_dt: Optional[datetime] = None) -> List[datetime]:
        """Kembalikan datetime JST untuk tiap jam posting pada tanggal reference."""
        if reference_dt is None:
            reference_dt = datetime.now(self.tz)
        today: date = reference_dt.date()
        slots: List[datetime] = []
        for h in sorted(set(self.posting_hours)):
            if not 0 <= h <= 23:
                continue
            naive = datetime.combine(today, time(hour=h, minute=0))
            slots.append(self.tz.localize(naive))
        return sorted(slots)

    def upcoming_slots(self, reference_dt: Optional[datetime] = None) -> List[datetime]:
        """Slot hari ini yang belum lewat."""
        now = reference_dt or datetime.now(self.tz)
        return [s for s in self.compute_schedule(now) if s > now]

    # ------------------------------------------------------------------
    # APScheduler
    # ------------------------------------------------------------------

    def start(self):
        """
        Mulai scheduler (blocking). Memasang satu cron job yang menyala pada
        tiap jam posting JST, sehingga jadwal otomatis berulang tiap hari.
        """
        from apscheduler.schedulers.blocking import BlockingScheduler

        self._scheduler = BlockingScheduler(timezone=self.tz)
        hours_str = ",".join(str(h) for h in sorted(set(self.posting_hours)))

        self._scheduler.add_job(
            self._run_cycle,
            trigger="cron",
            hour=hours_str,
            minute=0,
            id="content_cron",
            replace_existing=True,
            misfire_grace_time=300,  # 5 menit grace period
        )

        logger.info(
            "ContentScheduler dimulai: %d/hari | jam JST %s",
            self.daily_target, hours_str,
        )
        self._scheduler.start()

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
