"""Turn per-frame hand poses into gesture events.

- Fist held still: pause. Open palm, facing the camera, held still: play.
- Index finger pointing right / left, held: next / previous track.
- Open hand turned like a volume knob: clockwise turns the volume up, the other way turns it down.
- Middle finger alone, held: quit the app.
"""

from collections import deque

OPEN, FIST, OTHER, NONE = "open", "fist", "other", "none"
POINT_LEFT, POINT_RIGHT, MIDDLE = "point_left", "point_right", "middle_finger"

CLASSIFIER_MIN_SCORE = 0.5  # below this the classifier's label is not trusted on its own
OTHER_GESTURES = {"Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"}
FRONTAL_MIN = 0.55  # palm width / length below which an open palm is seen too much from the side to play


def pose_of(hand, frontal_min: float = FRONTAL_MIN) -> str:
    """One of OPEN, FIST, POINT_LEFT, POINT_RIGHT, MIDDLE, OTHER, or NONE (no hand).

    The hand's shape decides pointing and the middle finger (the trained classifier knows neither). Otherwise the
    classifier decides when it is confident; when it is not (it says "None", e.g. for a blurred hand),
    counting the raised fingers decides. An open palm counts as OPEN only when seen roughly from the front.
    """
    if hand is None:
        return NONE
    if hand.pointing:
        return POINT_RIGHT if hand.pointing == "right" else POINT_LEFT
    thumb, index, middle, ring, pinky = hand.fingers
    if middle and not (index or ring or pinky):  # the thumb may be either way
        return MIDDLE
    confident = hand.score >= CLASSIFIER_MIN_SCORE
    if confident and hand.gesture == "Closed_Fist":
        return FIST
    if confident and hand.gesture in OTHER_GESTURES:
        return OTHER
    if (confident and hand.gesture == "Open_Palm") or (hand.fingers_up >= 4 and hand.gesture != "Closed_Fist"):
        return OPEN if hand.frontal >= frontal_min else OTHER
    if hand.fingers_up == 0 and hand.gesture != "Open_Palm":
        return FIST
    return OTHER


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class PoseHold:
    """Emits an event when one of its poses is held still for `hold` seconds: by default "pause" for a fist
    and "play" for an open palm.

    - Once per hold: to fire again, change the pose or take the hand away.
    - The hand must stay in place: if its palm center moves more than `max_move` from where the hold began,
      the hold starts over from there.
    - Short misreadings do not break a hold: the hand lost, or seen in another pose, for up to `max_gap`
      seconds.
    """

    def __init__(self, events: dict | None = None, hold: float = 2.0, max_move: float = 0.15,
                 max_gap: float = 0.3):
        self.events = events or {FIST: "pause", OPEN: "play"}
        self.hold = hold
        self.max_move = max_move
        self.max_gap = max_gap
        self.progress: tuple[str, float] | None = None  # (event, 0..1 of the hold), for display
        self.blocked_until = -1e9  # no hold can finish before this; it starts counting from then
        self._reset()

    def restart(self):
        """Forget the hold under way (e.g. the hand is busy turning the volume)."""
        self._reset()

    def _reset(self):
        self._pose = None  # the pose being held
        self._since = self._seen = 0.0  # when its hold began, when it was last seen
        self._anchor = None  # where the hand was when the hold began
        self._fired = False
        self._candidate, self._candidate_since = None, 0.0  # another pose that may be replacing it
        self.progress = None

    def _start(self, pose, since, pos):
        self._pose, self._since, self._anchor, self._fired = pose, since, pos, False
        self._candidate = None

    def update(self, pose: str, now: float, pos: tuple[float, float] | None = None) -> str | None:
        """`pos` is the palm center (optional): with it, a moving hand does not count as held."""
        target = pose if pose in self.events else None
        if target is not None and target == self._pose:
            self._candidate = None
            if pos and self._anchor and _dist(pos, self._anchor) > self.max_move:
                self._start(target, now, pos)  # the hand moved: hold again from here
        elif target is not None:
            # A new pose. While the old one was seen within max_gap it may be a misreading; once it lasts
            # longer it is real, and its hold counts from when it appeared.
            if self._candidate != target:
                self._candidate, self._candidate_since = target, now
            if self._pose is not None and now - self._seen <= self.max_gap:
                return None
            self._start(target, self._candidate_since, pos)
        else:
            self._candidate = None
            if self._pose is not None and now - self._seen > self.max_gap:
                self._reset()  # gone too long
            return None  # no hand or another pose: tolerated for max_gap
        self._seen = now

        if self._fired:
            self.progress = None
            return None
        if now < self.blocked_until:
            self._since = now  # pause after another action: the hold counts from when it ends
            self.progress = None
            return None
        held = now - self._since
        event = self.events[self._pose]
        if held >= self.hold:
            self._fired = True
            self.progress = None
            return event
        self.progress = (event, held / self.hold)
        return None


class VolumeKnob:
    """The open hand as a volume knob that works like a ratchet.

    Where the hand is when it has been open for `settle` seconds is neutral. Turning away from neutral changes
    the volume: every `step_angle` degrees clockwise emits "volume_up", the other way "volume_down". Turning
    back toward neutral changes nothing, however slowly, so the hand can be brought back; once back near
    neutral, the knob is ready for another turn. A hand that comes into view already tilted starts its neutral
    there. Angles are smoothed over `smoothing` frames; short losses of the open hand (up to `max_gap` seconds)
    keep the knob.
    """

    def __init__(self, step_angle: float = 15.0, settle: float = 0.3, max_gap: float = 0.3, smoothing: int = 3,
                 busy_after: float = 2.0):
        self.step_angle = step_angle
        self.settle = settle
        self.max_gap = max_gap
        self.busy_after = busy_after
        self._raw = deque(maxlen=smoothing)
        self._last_event = -1e9
        self.reset()

    def reset(self):
        self._since = None  # when the open hand appeared
        self._seen = 0.0
        self._neutral = None
        self._steps = 0  # steps taken in this turn: + clockwise, - the other way
        self._turn = 0.0  # degrees from neutral
        self._raw.clear()
        self.progress: tuple[str, float] | None = None  # ("knob_up" / "knob_down", 0..1 of a step), for display

    def busy(self, now: float) -> bool:
        """The hand is turning the knob (or just did): it is not holding a sign."""
        turned = self._neutral is not None and abs(self._turn) > self.step_angle / 2
        return turned or now - self._last_event < self.busy_after

    def update(self, pose: str, now: float, angle: float | None) -> str | None:
        if pose != OPEN or angle is None:
            if self._since is not None and now - self._seen > self.max_gap:
                self.reset()
            return None
        self._seen = now
        if self._since is None:
            self._since = now
        self._raw.append(angle)
        a = sum(self._raw) / len(self._raw)
        if now - self._since < self.settle:
            return None
        if self._neutral is None:
            self._neutral = a
        turn = self._turn = a - self._neutral  # + clockwise
        step = self.step_angle

        if abs(turn) < step / 2:
            self._steps = 0  # back near neutral: ready for a new turn
        elif (self._steps > 0 and turn < 0) or (self._steps < 0 and turn > 0):
            self._steps = 0  # went past neutral to the other side: a turn that way starts

        # The next step is one more step away from neutral, on the side already being turned.
        if self._steps >= 0 and turn >= (self._steps + 1) * step:
            self._steps += 1
            event = "volume_up"
        elif self._steps <= 0 and turn <= (self._steps - 1) * step:
            self._steps -= 1
            event = "volume_down"
        else:
            toward = turn - self._steps * step  # degrees past the last step, on the side being turned
            if self._steps > 0 or (self._steps == 0 and turn > 0):
                self.progress = ("knob_up", max(0.0, min(1.0, toward / step)))
            else:
                self.progress = ("knob_down", max(0.0, min(1.0, -toward / step)))
            return None
        self._last_event = now
        self.progress = None
        return event
