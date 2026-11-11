"""user_logs domain: the daily ``(uid, date)`` listening table."""

from w2.src.data_processing.user_logs.daily import build_user_day_logs

__all__ = ["build_user_day_logs"]
