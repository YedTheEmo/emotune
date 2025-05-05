from FoxDot import *

# Set Tempo & Energy
Clock.bpm = 160  # Faster tempo for aggression

# **Drums - Hard Rock Beat**
d1 >> play("XxX- X-oX", sample=2, amp=1.2)  # Hard-hitting kick & snare
d2 >> play("---- [-x]- [-x]", amp=0.8)  # Hi-hats & cymbals
d3 >> play("--[--]--[**]", dur=4, amp=0.7)  # Crash every 4 bars

# **Bass - Driving Power**
b1 >> sawbass([0, 0, 3, 3, 5, 5, 3, 3], dur=1, amp=1.0, cutoff=700, lpf=900)

# **Rhythm Guitar - Power Chords**
g1 >> sawbass([0, 3, 5, 3], dur=1/2, amp=1.1, cutoff=800, lpf=900).every(4, "reverse")

# **Lead Guitar - Screaming Licks**
g2 >> zap([7, 7, 10, 12, 10, 7, 5, 3], dur=[1,1,1/2,1/2,1,1,1,1], amp=1.3, cutoff=1000).every(6, "offadd", 2)

# **Pre-Chorus Fill**
fill = play("  *  ", dur=1, amp=0.9).every(8, "stutter", 4)

# **Chorus Section - More Power**
def chorus():
    print("CHORUS")
    g1 >> sawbass([5, 7, 8, 7], dur=1/2, amp=1.3, cutoff=900, lpf=1000)
    g2 >> zap([12, 14, 15, 14, 12, 10, 8, 7], dur=1/2, amp=1.5).every(6, "offadd", 2)

# **Guitar Solo - Shred Mode**
def solo():
    print("SOLO TIME!")
    g1.stop()  # Kill rhythm guitar for solo space
    g2 >> zap([12, 15, 17, 15, 12, 10, 8, 7, 19, 17, 15, 12], dur=1/4, amp=1.7, pan=[-1,1], lpf=1200).every(4, "reverse")

# **Final Breakdown - Heavy Outro**
def outro():
    print("FINAL BREAKDOWN")
    g1 >> sawbass([0, 3, 5, 3], dur=1/2, amp=1.5, cutoff=700)
    g2 >> zap([3, 5, 7, 8, 10, 12, 15], dur=1/4, amp=1.8, pan=[-1,1]).every(3, "stutter", 2)

# **Song Structure**
def play_song():
    print("INTRO")
    Clock.future(8, lambda: print("VERSE 1"))
    Clock.future(16, chorus)  # First chorus
    Clock.future(32, lambda: print("VERSE 2"))
    Clock.future(48, chorus)  # Second chorus
    Clock.future(64, solo)  # Guitar Solo
    Clock.future(80, lambda: print("FINAL CHORUS"))
    Clock.future(96, outro)  # Heavy breakdown outro
    Clock.future(112, lambda: Clock.clear())  # Stop song

# **Start the Song**
play_song()

