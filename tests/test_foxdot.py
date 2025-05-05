from FoxDot import *

# Set tempo
Clock.bpm = 120  # Beats per minute

# Basic drum beat
d1 >> play("x-o-")  # 'x' = kick, 'o' = snare

# Hi-hats
d2 >> play("---[--]", sample=2)

# Bassline
b1 >> bass([0, 2, 3, 5], dur=1)

# Chords
p1 >> pads([0, 4, 5, 7], dur=4, amp=0.6)

# Melodic arpeggio
m1 >> pluck([0, 2, 4, 7], dur=[0.5, 0.5, 1, 1.5])


