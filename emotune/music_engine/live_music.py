
from FoxDot import *

Clock.bpm = 110
Scale.default = Scale.minor
Root.default = 4

p1 >> saw(
    [0, 2, 4, 5],
    dur=[1, 0.5, 1, 0.5],
    amp=0.7
)

d1 >> play("x-o-", sample=2, rate=1.2, amp=0.7333333333333333)
