from FoxDot import *
# Set Tempo
Clock.bpm = 120

# 🎵 Drums
d1 >> play("x-o- *-o-", sample=2, amp=0.8)  # Kick and snare
d2 >> play("--[--]--[--]", sample=5, amp=0.5)  # Hi-hat pattern
d3 >> play("  [--] [**]  ", dur=4, amp=0.6)  # Crash every 4 bars

# 🎸 Bass
b1 >> bass([0, 0, 3, 3, 5, 5, 3, 3], dur=1, amp=0.7).every(8, "stutter", 2)

# 🎸 Rhythm Guitar (Power Chords)
g1 >> sawbass([0, 3, 5, 3], dur=2, amp=0.8, cutoff=600, lpf=800).every(4, "reverse")

# 🎸 Lead Guitar (Melody / Solo)
g2 >> pluck([7, 7, 10, 12, 10, 7, 5, 3], dur=[1,1,1/2,1/2,1,1,1,1], amp=0.9).every(6, "offadd", 2)

# 🎵 Transition Fill (Every 8 bars)
fill = play("  *  ", dur=1, amp=0.7).every(8, "stutter", 4)

# 🎵 Chorus Section
def chorus():
    g1 >> sawbass([5, 7, 8, 7], dur=2, amp=0.9, cutoff=700, lpf=900)
    g2 >> pluck([12, 10, 8, 7, 10, 8, 7, 5], dur=1/2, amp=1.0).every(6, "offadd", 2)

# 🎵 Structure
def play_song():
    print("Verse 1")
    Clock.future(16, lambda: print("Chorus"))
    Clock.future(16, chorus)
    Clock.future(32, lambda: print("Verse 2"))
    Clock.future(48, lambda: print("Chorus"))
    Clock.future(64, lambda: print("Guitar Solo"))
    Clock.future(64, lambda: g2 >> pluck([12,14,15,14,12,10,8,7], dur=1/2, amp=1.2))
    Clock.future(80, lambda: print("Final Chorus"))
    Clock.future(96, lambda: Clock.clear())  # Stop the song

play_song()

