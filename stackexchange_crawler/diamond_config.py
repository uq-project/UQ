"""Example configs for diamond subset collection.

This module defines stricter filtering criteria for the diamond subset, which targets
the higher quality unanswered questions from Stack Exchange sites.

Adjust these values based on site-specific characteristics and quality requirements.
"""

# Default diamond mode settings (conservative/high-quality); edit as needed
DEFAULT_DIAMOND_SETTINGS = {
    "min_score": 50,           # Minimum upvotes
    "min_views": 1000,        # Minimum views
    "min_age_days": 365 * 2,  # 2 years old
    "top_percentage": 5.0     # Top 5% of questions
}

# Site-specific overrides for diamond mode
# These account for different community sizes and engagement patterns
SITE_SPECIFIC_DIAMOND_SETTINGS = {
    "cstheory": {
        "min_score": 20,        # Smaller community, lower thresholds
        "min_views": 500
    },
    "math": {
        "min_score": 75,        # Large active community, higher thresholds
        "min_views": 2000,
    },
    "mathoverflow.net": {
        "min_score": 50,        # Large active community, higher thresholds
        "min_views": 2000,
    },
    "physics": {
        "min_score": 20,        # Smaller community, lower thresholds
        "min_views": 500,
    },
    "puzzling": {
        "min_score": 20,
        "min_views": 1000,
    },
    "scifi": {
        "min_score": 30,
        "min_views": 500,
    },
    "stackoverflow.com": {
        "min_score": 50,        # Massive community, many low-quality questions
        "min_views": 2000,
    },
    "tex": {
        "min_score": 20,
        "min_views": 500,
    },
}


def get_diamond_settings(site_name):
    """
    Get the diamond settings for a specific site.

    Args:
        site_name: Stack Exchange site API parameter (e.g., 'math', 'physics')

    Returns:
        A dictionary containing the diamond settings for the site,
        using default values for any settings not specifically overridden.
    """
    # Start with default settings
    settings = DEFAULT_DIAMOND_SETTINGS.copy()

    # Apply site-specific overrides if available
    if site_name in SITE_SPECIFIC_DIAMOND_SETTINGS:
        settings.update(SITE_SPECIFIC_DIAMOND_SETTINGS[site_name])

    return settings
