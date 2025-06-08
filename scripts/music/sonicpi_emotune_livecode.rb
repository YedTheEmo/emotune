# Sonic Pi Live Coding Environment for Emotion-to-Music Cue-Based System
# Copy and paste this into your Sonic Pi editor and run it before starting your Python backend.

# --- Pattern and Parameter Definitions ---

# Example: Define several musical patterns and parameterized behaviors
patterns = [
  [:c4, :e4, :g4],    # Pattern 0: C major chord
  [:d4, :f4, :a4],    # Pattern 1: D minor chord
  [:e4, :g4, :b4],    # Pattern 2: E minor chord
  [:f4, :a4, :c5],    # Pattern 3: F major chord
  [:g4, :b4, :d5]     # Pattern 4: G major chord
]

# Default parameters (will be updated by OSC cues)
set :current_pattern, 0
set :tempo, 120
set :overall_volume, 0.7
set :voice_density, 2
set :brightness, 0.5
set :reverb_amount, 0.3
set :articulation, 0.5
set :rhythm_complexity, 0.5
set :dissonance_level, 0.2

# --- OSC Listeners for Parameter Cues ---
live_loop :pattern_select do
  use_real_time
  idx = sync "/osc*/pattern_select"
  set :current_pattern, idx[0]
end

live_loop :tempo_control do
  use_real_time
  t = sync "/osc*/tempo"
  set :tempo, t[0]
end

live_loop :volume_control do
  use_real_time
  v = sync "/osc*/overall_volume"
  set :overall_volume, v[0]
end

live_loop :voice_density_control do
  use_real_time
  vd = sync "/osc*/voice_density"
  set :voice_density, vd[0]
end

live_loop :brightness_control do
  use_real_time
  b = sync "/osc*/brightness"
  set :brightness, b[0]
end

live_loop :reverb_control do
  use_real_time
  r = sync "/osc*/reverb_amount"
  set :reverb_amount, r[0]
end

live_loop :articulation_control do
  use_real_time
  a = sync "/osc*/articulation"
  set :articulation, a[0]
end

live_loop :rhythm_complexity_control do
  use_real_time
  rc = sync "/osc*/rhythm_complexity"
  set :rhythm_complexity, rc[0]
end

live_loop :dissonance_control do
  use_real_time
  d = sync "/osc*/dissonance_level"
  set :dissonance_level, d[0]
end

# --- Main Music Loop ---
live_loop :emotion_music do
  use_real_time
  use_bpm get(:tempo)
  pattern = patterns[get(:current_pattern) % patterns.length]
  density = get(:voice_density).to_i
  vol = get(:overall_volume)
  bright = get(:brightness)
  reverb_amt = get(:reverb_amount)
  art = get(:articulation)
  rhythm = get(:rhythm_complexity)
  diss = get(:dissonance_level)

  with_fx :reverb, mix: reverb_amt do
    density density do
      pattern.each do |n|
        use_synth :piano
        play n, amp: vol, attack: art * 0.2, release: (1 - art) * 0.5 + 0.2, cutoff: 70 + bright * 50, pan: rrand(-0.5, 0.5),
          sustain: 0.2 + rhythm * 0.5, detune: diss * 10
        sleep 0.5 / (1 + rhythm)
      end
    end
  end
  sleep 2
end

# --- End of Sonic Pi Live Code ---
# You can add more live_loops for melody, bass, drums, etc., using the same cue-based approach.
