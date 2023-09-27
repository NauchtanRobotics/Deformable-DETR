def in_colab():
    """Returns True if running in Google Colaboratory."""
    try:
        from google import colab
        return True
    except ImportError:
        return False