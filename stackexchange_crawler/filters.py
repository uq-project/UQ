"""Question quality filters for Stack Exchange crawler.

This module implements rule-based filtering to identify high-quality,
unanswered questions from Stack Exchange sites. Filters include:
- Engagement metrics (score, views, age)
- Content quality (avoiding images, recommendation requests)
- Site-specific criteria
"""

from datetime import datetime, timedelta
import re
from typing import Dict, Callable
from loguru import logger
import argparse
import json
import os

# Quality thresholds
MAX_VIEW_TO_VOTE_RATIO = 5000  # Avoid questions with many views but few upvotes

# Content filtering patterns
IMAGE_PATTERN = re.compile(r'!\[.*?\]\(.*?\)|<img.*?src=.*?>|!\[.*?\]\[.*?\]')  # Detect images in markdown/HTML

WHY_PATTERN = re.compile(r'\bwhy\b', re.IGNORECASE)  # "Why" questions tend to be subjective

# Tags that indicate low-quality or off-topic questions (exact matches)
EXACT_FILTERED_TAGS = {
    # Recommendation requests
    'recommendation', 'recommendations', 'reference-request', 'book-recommendation',
    'tool-recommendation', 'software-recommendation', 'product-recommendation', 'shopping', 'list',

    # Subjective/opinion-based
    'discussion', 'opinion', 'poll', 'subjective', 'debate', 'open-ended', 'argumentative',

    # Educational/homework
    'homework', 'assignment', 'exam', 'test', 'quiz', 'problem-set', 'school', 'class',

    # Career/personal advice
    'career-development', 'career-advice', 'job-search', 'interview', 'resume', 'cv', 'salary',
    'workplace', 'personal',

    # Site maintenance/meta
    'off-topic', 'meta', 'faq', 'support', 'bug', 'feature-request', 'site-policy',
    'moderation', 'announcement', 'duplicate', 'closed', 'unclear', 'too-broad',
    'not-constructive', 'invalid', 'wont-fix',

    # Soft/social questions
    'soft-question', 'fun', 'joke', 'community-wiki', 'survey'
}

# Keywords for substring matching in tags (catch variations)
SUBSTRING_KEYWORDS = [
    'recommend', 'resource', 'book', 'shopping', 'opinion', 'poll', 'debate',
    'homework', 'assignment', 'exam', 'career', 'job', 'interview', 'salary',
    'off-topic', 'meta', 'bug', 'site-policy', 'duplicate', 'unclear', 'soft', 'fun', 'joke'
]


def filter_check(question_data: dict, site: str, min_score: int, min_age_days: int,
                 min_views: int) -> tuple[bool, str]:
    """Apply quality filters to determine if a question should be included.

    Args:
        question_data: Dictionary containing question details from Stack Exchange API
        site: Stack Exchange site name (e.g., 'math', 'physics')
        min_score: Minimum upvote score threshold
        min_age_days: Minimum age in days (ensures question has had time to be answered)
        min_views: Minimum view count (indicates community interest)

    Returns:
        Tuple of (passes_filters: bool, reason: str)
        - True if question passes all filters, False otherwise
        - Reason string explains why question was filtered out
    """
    if not question_data:
        return False, "Empty question data"

    question_id = question_data.get('id')
    views = question_data.get('views', 0)
    votes = question_data.get('score', 0)

    # --- Engagement and Quality Metrics ---
    # Check votes
    if votes < min_score:
        reason = f"Score {votes} < {min_score}"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # Check minimum age (older questions more likely to be genuinely hard)
    try:
        creation_dt = datetime.fromisoformat(question_data['creation_date'])
        if creation_dt > (datetime.now() - timedelta(days=min_age_days)):
            reason = f"Age is less than {min_age_days} days"
            logger.debug(f"[{site}] QID {question_id}: {reason}.")
            return False, reason
    except (ValueError, TypeError) as e:
        reason = f"Invalid creation date '{question_data.get('creation_date')}': {e}"
        logger.warning(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # Filter out questions with high view-to-vote ratio (likely low quality)
    if votes > 0 and views > 0 and (views / votes) > MAX_VIEW_TO_VOTE_RATIO:
        reason = f"Poor view-to-vote ratio, {views}/{votes} = {views/votes:.1f}"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # Ensure minimum community interest
    if views < min_views:
        reason = f"Insufficient views ({views} < {min_views})"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # --- Content Quality Checks ---
    # Filter out questions with images (focus on text-based questions)
    body = question_data.get('body', '')
    body_markdown = question_data.get('body_markdown', '')
    if IMAGE_PATTERN.search(body_markdown) or IMAGE_PATTERN.search(body):
        reason = "Contains images"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # Filter subjective "why" questions
    title = question_data.get('title', '')
    if WHY_PATTERN.search(title):
        reason = "'Why' question"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # --- Tag-based Filtering ---
    # Check for problematic tags (exact matches)
    lower_tags = [tag.lower() for tag in question_data.get('tags', [])]
    tags_set = set(lower_tags)
    if not EXACT_FILTERED_TAGS.isdisjoint(tags_set):
        filtered_tags = EXACT_FILTERED_TAGS.intersection(tags_set)
        reason = f"Contains filtered tags: {list(filtered_tags)}"
        logger.debug(f"[{site}] QID {question_id}: {reason}")
        return False, reason

    # Check for problematic tag patterns (substring matches)
    for tag in lower_tags:
        if any(keyword in tag for keyword in SUBSTRING_KEYWORDS):
            reason = f"Tag '{tag}' matches filtered pattern"
            logger.debug(f"[{site}] QID {question_id}: {reason}")
            return False, reason

    # --- Site-Specific Quality Checks ---
    if site_specific_checks.get(site):
        if not site_specific_checks[site](question_data):
            reason = "Failed site-specific checks"
            logger.debug(f"[{site}] QID {question_id}: {reason}")
            return False, reason

    return True, "All filters passed"


# --------- Site-specific validation functions ---------


def math_checks(question_data: dict) -> bool:
    """Site-specific checks for mathematics sites."""
    # Specific checks for math.stackexchange.com and mathoverflow.net
    tags = set(tag.lower() for tag in question_data.get('tags', []))
    # Filter out homework questions and soft questions
    FILTERED_MATH_TAGS = {'homework', 'soft-question'}
    if not FILTERED_MATH_TAGS.isdisjoint(tags):
        return False
    return True


def programming_checks(question_data: dict) -> bool:
    """Site-specific checks for programming sites."""
    # For sites like Stack Overflow, CS Theory, etc.
    tags = set(tag.lower() for tag in question_data.get('tags', []))
    # Filter out career advice
    FILTERED_PROGRAMMING_TAGS = {'career-development', 'career-advice'}
    if not FILTERED_PROGRAMMING_TAGS.isdisjoint(tags):
        return False

    title = question_data.get('title', '').lower()
    # Filter out "best practice" questions which tend to be opinion-based
    FILTERED_TITLE_PHRASES = {'best practice', 'best way'}
    if any(phrase in title for phrase in FILTERED_TITLE_PHRASES):
        return False

    return True


def physics_checks(question_data: dict) -> bool:
    """Site-specific checks for physics site."""
    tags = set(tag.lower() for tag in question_data.get('tags', []))

    # Filter out homework-like questions and resource recommendations
    FILTERED_PHYSICS_TAGS = {'homework', 'recommendations'}
    if not FILTERED_PHYSICS_TAGS.isdisjoint(tags):
        return False

    return True


# --- Site-Specific Filter Registry ---
# Maps site API parameters to their specific validation functions
# Site names correspond to api_site_parameter in sites.jsonl
site_specific_checks: Dict[str, Callable[[dict], bool]] = {
    # Math sites
    'math': math_checks,
    'mathoverflow.net': math_checks,
    'stats': math_checks,
    'cstheory': math_checks,

    # Programming sites
    'cs': programming_checks,
    'stackoverflow': programming_checks,
    'codereview': programming_checks,
    'tex': programming_checks,

    # Physics site
    'physics': physics_checks,

    # Add more sites here
}


def main():
    """Standalone filter application for processing existing question datasets.

    Reads a JSONL file of questions, applies all filters, and saves results.
    Useful for post-processing or testing filter configurations.
    """
    args = parse_args()

    logger.info(f"Running filter with min_score={args.min_score}, min_age_days={args.min_age_days}")
    logger.info(f"Reading from {args.input_file}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filtered_count = 0
    total_count = 0

    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total_count += 1
            try:
                question_data = json.loads(line.strip())
                filter_result, reason = filter_check(question_data, args.min_score,
                                                     args.min_age_days, args.min_views)
                if filter_result:
                    outfile.write(line)
                    filtered_count += 1
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line: {e}")

    logger.info(
        f"Completed static filtering. Filtered {filtered_count} out of {total_count} questions.")
    logger.info(f"Results saved to {args.output_file}")


# --- Command Line Interface ---
def parse_args():
    """Parse command line arguments for running static, rule-based filters."""
    parser = argparse.ArgumentParser(description="Static question filters")
    parser.add_argument('--min_score',
                        type=int,
                        default=5,
                        help='Minimum score for a question to be considered')
    parser.add_argument('--min_age_days',
                        type=int,
                        default=365 * 2,
                        help='Minimum age in days for a question')
    parser.add_argument('--min_views', type=int, default=100, help='Minimum views for a question')
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Input JSONL file containing questions')
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='Output JSONL file for filtered questions')
    return parser.parse_args()


if __name__ == "__main__":
    main()
