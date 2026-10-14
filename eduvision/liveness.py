"""Liveness detection (anti-spoofing). Planned for Week 2.

Right now every face is treated as live, which means a printed photo or a
phone screen can fool the system. Week 2: record that weakness first, then
add blink detection here and measure the improvement.
"""


class LivenessChecker:
    enabled = False

    def update(self, frame, face) -> None:
        """Feed each new frame for the tracked face (unused for now)."""

    def is_live(self) -> bool:
        return True
