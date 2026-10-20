"""Tiny progress reporter shared by rollout.py and paper_eval.py (stdlib only).

Adapts to where the output goes:
  * a terminal  -> one line rewritten in place with '\r'
  * a pipe/file -> a fresh line every `interval` seconds, so `| tee run.log` stays readable

Always prints the final line, and never emits more than one update per `interval`.
"""
import shutil
import sys
import time


def fmt_hms(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "--:--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Progress:
    def __init__(self, total, label, interval=5.0, stream=None):
        self.total, self.label, self.interval = total, label, interval
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.t0 = time.time()
        self.last = 0.0
        self.done = 0

    def _line(self, done, extra):
        el = time.time() - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 and self.total else None
        pct = 100.0 * done / self.total if self.total else 0.0
        bar = ""
        if self.tty:
            width = max(10, min(28, shutil.get_terminal_size((100, 20)).columns - 78))
            filled = int(width * done / self.total) if self.total else 0
            bar = " [" + "#" * filled + "-" * (width - filled) + "]"
        rate_s = f"{rate*60:.1f}/min" if rate < 1 else f"{rate:.1f}/s"
        return (f"[{self.label}]{bar} {done}/{self.total} ({pct:.0f}%) | {extra}"
                f"{' | ' if extra else ''}{rate_s} | elapsed {fmt_hms(el)} | ETA {fmt_hms(eta)}")

    def update(self, done, extra="", force=False):
        self.done = done
        now = time.time()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        line = self._line(done, extra)
        if self.tty:
            self.stream.write("\r\033[K" + line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self, extra=""):
        self.update(self.done, extra, force=True)
        if self.tty:
            self.stream.write("\n")
        self.stream.flush()
