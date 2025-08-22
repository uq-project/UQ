"""One-off Stack Exchange question crawler.

This module provides functionality to crawl individual Stack Exchange questions
by URL and extract their content and metadata.
"""

import argparse
import json
import os
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Tuple
from dotenv import load_dotenv

# --- Load Stack Exchange API key (strongly recommended) ---
# Register at https://api.stackexchange.com/ to get your free API key
# Place it in a key.env file as: STACKEXCHANGE_API_KEY=your_key_here
load_dotenv("key.env")
load_dotenv("../key.env")
STACKEXCHANGE_API_KEY = os.getenv("STACKEXCHANGE_API_KEY")


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    if not html or not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, 'html.parser', on_duplicate_attribute='ignore')
    return soup.get_text()


def parse_stackexchange_url(url: str) -> Tuple[str, str]:
    """Extract site name and question ID from URL."""
    site_pattern = r'https?://([^/]+)'
    id_pattern = r'/questions/(\d+)'

    site = re.findall(site_pattern, url)[0].split('.')[0]
    question_id = re.findall(id_pattern, url)[0]
    return site, question_id


def crawl_stackexchange(url: str) -> Tuple[str, Dict]:
    """Crawl a Stack Exchange question and return formatted text and data dict."""
    site, question_id = parse_stackexchange_url(url)
    return crawl_stackexchange_by_id(site, question_id)


def crawl_stackexchange_by_id(site: str, question_id: str) -> Tuple[str, Dict]:
    """Crawl a Stack Exchange question by ID and return formatted content.

    Args:
        site: The Stack Exchange site name (e.g., 'math', 'physics')
        question_id: The Stack Exchange question ID

    Returns:
        Tuple containing:
        - formatted_text: Human-readable question and answer text
        - metadata: Dictionary with structured question data
    """
    # Construct human accessible URL
    url = f"https://{site}.stackexchange.com/questions/{question_id}"

    # Fetch question data
    question_url = f"https://api.stackexchange.com/2.3/questions/{question_id}"
    params = {"order": "desc", "sort": "activity", "site": site, "filter": "withbody"}

    # Add API key if available
    if STACKEXCHANGE_API_KEY:
        params["key"] = STACKEXCHANGE_API_KEY

    try:
        response = requests.get(question_url, params=params)
        question_data = response.json()['items'][0]
    except Exception as e:
        print(f"API request failed (possibly throttled): {e}")
        return "", {}

    # Fetch answers sorted by votes
    answer_url = f"{question_url}/answers"
    params.update({"sort": "votes"})
    answers_data = requests.get(answer_url, params=params).json()['items']

    # Process answers - prefer accepted answer, fallback to highest voted
    answer_text = "[Unanswered]"
    if answers_data:
        # Get accepted answer or highest voted answer (first in list since sorted by votes)
        answer_data = next((a for a in answers_data if a.get('is_accepted')), answers_data[0])
        answer_text = html_to_text(answer_data['body'])

    # Fetch comments sorted chronologically
    comments_url = f"https://api.stackexchange.com/2.3/questions/{question_id}/comments"
    params.update({"sort": "creation"})
    comments_data = requests.get(comments_url, params=params).json()['items']

    # Extract comment text
    comments = [html_to_text(comment['body']) for comment in comments_data]

    # Format text output
    text_output = f"""
------- BEGIN RAW TEXT QUESTION -------
{html_to_text(question_data['body'])}
------- END RAW TEXT QUESTION -------

------- BEGIN RAW TEXT ANSWER -------
{answer_text}
------- END RAW TEXT ANSWER -------
"""

    # Prepare structured metadata
    metadata = {
        "id": question_id,
        "site": site,
        "category": site,  # Site serves as category
        "link": url,
        "title": question_data['title'],
        "body": html_to_text(question_data['body']),
        "answer": answer_text,
        "comments": comments,
        "tags": question_data['tags']
    }

    return text_output, metadata


def main():
    """Command line interface for the one-off crawler.

    Usage:
        python oneoff_crawler.py <stack_exchange_url>

    Example:
        python oneoff_crawler.py https://math.stackexchange.com/questions/358423/a-proof-of-dimrt-dimr1-without-prime-ideals
    """
    parser = argparse.ArgumentParser(
        description='Crawl a single Stack Exchange question and extract its content and metadata.')
    parser.add_argument(
        'url',
        type=str,
        help=
        'Stack Exchange question URL (e.g., https://math.stackexchange.com/questions/123456/...)')

    args = parser.parse_args()
    text_output, metadata = crawl_stackexchange(args.url)

    # Output results
    print("============== QUESTION METADATA (JSON) ==============")
    print(json.dumps(metadata, indent=2))

    print("\n============== FORMATTED TEXT OUTPUT ==============")
    print(text_output)


if __name__ == "__main__":
    main()
