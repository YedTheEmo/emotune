def snake_to_camel(s: str) -> str:
    """Convert snake_case string to camelCase."""
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])
