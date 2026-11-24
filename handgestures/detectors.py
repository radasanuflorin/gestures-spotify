"""All the gesture detectors together: held signs (play, pause, track, quit) and the volume knob.

Feed it one hand per camera frame (or None) and it returns the gestures that fired: "play", "pause", "next",
"previous", "volume_up", "volume_down", "quit".
"""

from .gestures import MIDDLE, NONE, POINT_LEFT, POINT_RIGHT, PoseHold, VolumeKnob, pose_of

KNOB_SHOWN_FROM = 0.15  # the knob's progress is shown once the hand has turned this much of a step
HAND_SWITCH_JUMP = 0.2  # with a change of handedness, a jump this big means the tracker moved to another hand
MAX_FRAME_GAP = 0.25  # a repeated image is still processed after this long, so the preview never freezes
TRAIL_SECONDS = 0.6  # how much of the palm's path the preview draws


def in_ignored_zone(center, ignore_bottom: float) -> bool:
    return center[1] > 1 - ignore_bottom


def usable_position(center, ignore_bottom: float) -> tuple[float, float]:
    """The palm center in the part of the image above the ignored zone, stretched to 0..1, so the gesture
    distances are fractions of the area you actually use."""
    return center[0], center[1] / (1 - ignore_bottom)


class Detectors:
    """Play/pause and track (held signs) and the volume knob, rebuilt to forget half-done gestures."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self.reset()

    def reset(self):
        self._last_hand = None
        self.ignored = False  # the last hand was in the ignored bottom zone
        self._quiet_until = -1e9  # the pause after an action
        self._last_event = None
        self.pose = NONE  # the last pose, for the preview
        pp, tr, vol = self._cfg["play_pause"], self._cfg["track"], self._cfg["volume"]
        self.play_pause = PoseHold(hold=pp["hold"], max_move=pp["max_move"])
        self.track = PoseHold({POINT_RIGHT: "next", POINT_LEFT: "previous"}, hold=tr["hold"],
                              max_move=tr["max_move"])
        self.volume = VolumeKnob(step_angle=vol["step_angle"], settle=vol["settle"])
        quit_cfg = self._cfg["quit"]
        self.quit = PoseHold({MIDDLE: "quit"}, hold=quit_cfg["hold"]) if quit_cfg["enabled"] else None

    @property
    def progress(self) -> tuple[str, float] | None:
        """The gesture under way, for the preview: the volume knob once it is really turning, else a sign being
        held (the knob reports 0% for any open hand, which would hide the bar for play)."""
        knob = self.volume.progress
        if knob and knob[1] >= KNOB_SHOWN_FROM:
            return knob
        return self.play_pause.progress or self.track.progress or (self.quit and self.quit.progress)

    def update(self, hand, now: float) -> list[str]:
        self.ignored = bool(hand) and in_ignored_zone(hand.center, self._cfg["hand"]["ignore_bottom"])
        if self.ignored:
            hand = None  # resting on the mouse or keyboard: not a gesture
        if self._switched_hand(hand):
            self.volume.reset()  # the knob's angle belonged to the other hand
        if hand:
            self._last_hand = hand
        pose = self.pose = pose_of(hand, self._cfg["play_pause"]["frontal"])
        center = usable_position(hand.center, self._cfg["hand"]["ignore_bottom"]) if hand else None
        volume = self.volume.update(pose, now, hand.angle if hand else None)
        if self.volume.busy(now):
            self.play_pause.restart()  # an open hand turning the knob is not holding the sign for play
        events = (self.play_pause.update(pose, now, center), self.track.update(pose, now, center), volume,
                  self.quit.update(pose, now, center) if self.quit else None)
        return self._after_cooldown([e for e in events if e], now)

    def _after_cooldown(self, events: list[str], now: float) -> list[str]:
        """After an action, nothing new fires for a moment, so the hand can be brought back. Only more volume
        steps the same way pass (one turn of the knob)."""
        passed = []
        for e in events:
            continuing_turn = e == self._last_event and e in ("volume_up", "volume_down")
            if now < self._quiet_until and not continuing_turn:
                continue
            passed.append(e)
            self._last_event = e
            self._quiet_until = now + self._cfg["actions"]["cooldown"]
            for hold in (self.play_pause, self.track, self.quit):
                if hold:
                    hold.blocked_until = self._quiet_until
        return passed

    def _switched_hand(self, hand) -> bool:
        """With two hands in view the tracker can jump from one to the other: the palm center then leaps
        across the image, and its angle with it, which would turn the volume knob. A jump together with a change of handedness
        is a switch; a handedness misread alone (the hand did not move far) is not."""
        last = self._last_hand
        if not hand or not last or not hand.handedness or not last.handedness or hand.handedness == last.handedness:
            return False
        dx, dy = hand.center[0] - last.center[0], hand.center[1] - last.center[1]
        return (dx * dx + dy * dy) ** 0.5 > HAND_SWITCH_JUMP
