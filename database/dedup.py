"""
database/dedup.py — Dedup semantik via sentence-transformers.

Cara kerja:
  1. Hitung embedding teks (model multilingual ringan)
  2. Bandingkan cosine similarity vs embedding konten 30 hari terakhir
  3. Jika similarity > threshold → duplikat, konten dibuang

Fungsi utama yang dipakai orchestrator:
  is_duplicate(text)         — cek saja, tidak simpan
  check_and_save(tweet_id, text) — cek LALU simpan embedding jika lolos
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger("twitter_ai.dedup")

# Singleton model — di-load sekali, dipakai ulang
_model = None


def _get_model():
    """Lazy-load embedding model. Download otomatis saat pertama kali."""
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers belum terinstall. "
            "Jalankan: pip install sentence-transformers\n"
            "Dedup semantik dinonaktifkan sementara."
        )
        return None

    from core.config import load_config
    model_name = load_config()["dedup"]["model"]
    logger.info("Memuat embedding model: %s (pertama kali mungkin lambat)", model_name)
    _model = SentenceTransformer(model_name)
    logger.info("Embedding model siap.")
    return _model


def compute_embedding(text: str) -> Optional[bytes]:
    """
    Hitung embedding teks, kembalikan sebagai bytes untuk disimpan ke DB.
    Kembalikan None jika model tidak tersedia.
    """
    import numpy as np

    model = _get_model()
    if model is None:
        return None

    # normalize_embeddings=True → cosine similarity = dot product (lebih cepat)
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32).tobytes()


def bytes_to_vector(blob: bytes):
    """Konversi bytes BLOB dari DB kembali ke numpy array."""
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a, b) -> float:
    """
    Cosine similarity antara dua vektor yang sudah ternormalisasi.
    Normalized → cukup dot product, lebih efisien.
    """
    import numpy as np
    return float(np.dot(a, b))


def _max_similarity(new_vec, recent_embeddings) -> Tuple[float, int]:
    """
    Cari similarity tertinggi antara new_vec dan semua embedding lama.
    Kembalikan (max_sim, tweet_id_paling_mirip).
    """
    max_sim = 0.0
    closest_id = -1
    for tweet_id, blob in recent_embeddings:
        sim = cosine_similarity(new_vec, bytes_to_vector(blob))
        if sim > max_sim:
            max_sim = sim
            closest_id = tweet_id
    return max_sim, closest_id


def is_duplicate(
    text: str,
    lookback_days: Optional[int] = None,
    threshold: Optional[float] = None,
) -> bool:
    """
    Cek apakah text terlalu mirip dengan konten yang sudah ada.
    Kembalikan True jika duplikat.

    Args:
        text: teks yang akan dicek (topik atau draft tweet)
        lookback_days: berapa hari ke belakang yang dicek (default dari config)
        threshold: ambang cosine similarity (default dari config)
    """
    from core.config import load_config
    from database.db_manager import get_recent_embeddings

    cfg = load_config()["dedup"]
    lookback_days = lookback_days or cfg["lookback_days"]
    threshold = threshold or cfg["similarity_threshold"]

    model = _get_model()
    if model is None:
        # Model tidak tersedia, dedup dilewati (aman — false negative, bukan false positive)
        return False

    recent = get_recent_embeddings(lookback_days)
    if not recent:
        return False  # belum ada history

    import numpy as np
    new_vec = np.frombuffer(compute_embedding(text), dtype=np.float32)
    max_sim, closest_id = _max_similarity(new_vec, recent)

    if max_sim > threshold:
        logger.info(
            "Duplikat terdeteksi: similarity=%.3f (threshold=%.2f) vs tweet #%d",
            max_sim, threshold, closest_id,
        )
        return True

    logger.debug("Dedup OK: max_similarity=%.3f < %.2f", max_sim, threshold)
    return False


def check_and_save(
    tweet_id: int,
    text: str,
    lookback_days: Optional[int] = None,
    threshold: Optional[float] = None,
) -> bool:
    """
    Cek duplikat LALU simpan embedding ke DB jika konten lolos.
    Panggil ini setelah tweet_id sudah ada di DB (setelah save_tweet_full).

    Returns:
        True  → lolos (bukan duplikat), embedding tersimpan
        False → duplikat, embedding TIDAK disimpan
    """
    from core.config import load_config
    from database.db_manager import get_recent_embeddings, save_embedding

    cfg = load_config()["dedup"]
    lookback_days = lookback_days or cfg["lookback_days"]
    threshold = threshold or cfg["similarity_threshold"]

    model = _get_model()
    if model is None:
        logger.warning("Dedup dilewati (model tidak tersedia), tweet #%d langsung lolos", tweet_id)
        return True

    embedding_bytes = compute_embedding(text)
    if embedding_bytes is None:
        return True  # gagal hitung embedding, anggap lolos

    import numpy as np
    new_vec = np.frombuffer(embedding_bytes, dtype=np.float32)

    recent = get_recent_embeddings(lookback_days)
    if recent:
        max_sim, closest_id = _max_similarity(new_vec, recent)
        if max_sim > threshold:
            logger.info(
                "Duplikat: similarity=%.3f vs tweet #%d → konten dibuang",
                max_sim, closest_id,
            )
            return False

    # Lolos dedup → simpan embedding untuk dicek oleh konten berikutnya
    save_embedding(tweet_id, embedding_bytes)
    logger.debug("Dedup OK, embedding disimpan untuk tweet #%d", tweet_id)
    return True
