
def generate_tidal_pattern(params: dict) -> str:
    """Generate a TidalCycles pattern string from music parameters."""
    # Example: Use a simple drum pattern and map some parameters
    gain = params.get('overall_volume', 0.7)
    speed = 1.0 + (params.get('rhythm_complexity', 0.5) - 0.5) * 0.5
    density = int(params.get('voice_density', 3))
    reverb = params.get('reverb_amount', 0.3)
    pattern = f'sound "bd sn" # speed {speed:.2f} # gain {gain:.2f} # density {density} # room {reverb:.2f}'
    return pattern
