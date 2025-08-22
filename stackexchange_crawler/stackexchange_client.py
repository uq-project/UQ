"""Stack Exchange API client for question crawling.

This module provides a robust client for interacting with the Stack Exchange API
to find and retrieve unanswered questions. Features include:
- Question discovery with ranking and filtering
- Question detail extraction
- Automatic retry with exponential backoff
- Rate limiting and quota management
"""

import requests
import time
import html2text
from loguru import logger
from datetime import datetime, timedelta
from typing import Dict, Tuple, Generator, Optional, List
from bs4 import BeautifulSoup

# API configuration
MIN_RETRY_DELAY = 5  # Minimum delay between retries (seconds)
ASSUMED_QUOTA_REMAINING = 50  # Fallback quota value when not provided by API


def should_retry_exception(exception):
    """Return True if the exception should trigger a retry."""
    # Retry on connection errors, timeouts, and specific API signals
    if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    # Retry on server-side errors (5xx) and potentially rate limits (429)
    if isinstance(exception, requests.exceptions.HTTPError):
        # Check if response exists before accessing status_code
        if exception.response is not None:
            return exception.response.status_code >= 500 or exception.response.status_code == 429
    return False


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    if not html or not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, 'html.parser', on_duplicate_attribute='ignore')
    return soup.get_text()


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown."""
    if not html or not isinstance(html, str):
        return ""
    converter = html2text.HTML2Text()
    converter.body_width = 0  # Disable line wrapping
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_emphasis = False
    return converter.handle(html)


class StackExchangeClient:
    """Handles interaction with the Stack Exchange API, tailored for finding unanswered questions.

    Includes methods for finding candidate questions based on criteria and fetching
    detailed information for specific question IDs.
    Utilizes the tenacity library for robust retry logic.
    """
    BASE_URL = "https://api.stackexchange.com/2.3"

    def __init__(self, api_key=None):
        """Initialize the Stack Exchange API client with optional API key.

        Args:
            api_key: Optional Stack Exchange API key for increased rate limits
                     (10,000 requests/day vs ~300 without key)
        """
        self.api_key = api_key
        self.session = requests.Session()
        logger.info(f"StackExchangeClient initialized. API Key Present: {bool(api_key)}")

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make authenticated API request with error handling.

        Args:
            endpoint: API endpoint path (e.g., 'questions/no-answers')
            params: Request parameters dictionary

        Returns:
            API response data dictionary

        Raises:
            ValueError: If required 'site' parameter is missing
            requests.HTTPError: For HTTP error responses
        """
        request_params = params.copy()
        if self.api_key:
            request_params['key'] = self.api_key
        # Ensure 'site' parameter is present, critical for SE API
        site = request_params.get('site')
        if not site:
            raise ValueError("'site' parameter is required for Stack Exchange API calls")

        url = f"{self.BASE_URL}/{endpoint}"
        logger.debug(f"Making SE API request: {url} with params: {request_params}")

        response = self.session.get(url, params=request_params)
        response.raise_for_status()
        data = response.json()
        return data or {}

    def find_potential_questions(self,
                                 site: str,
                                 min_score: int,
                                 min_age_days: int,
                                 page_size: int = 100,
                                 top_percentage: float = 10.0) -> Generator[Tuple[dict, int], None, None]:
        """Discover high-quality unanswered questions matching specified criteria.

        This method implements a two-phase approach:
        1. Determine the total pool of unanswered questions on the site
        2. Retrieve the top X% by vote count, applying filters for age and score

        Questions are returned in descending order by vote count.

        Args:
            site: Stack Exchange site name (e.g., 'math', 'physics', 'mathoverflow.net')
            min_score: Minimum upvote threshold
            min_age_days: Minimum age in days
            page_size: API results per page (max 100, default 100)
            top_percentage: Percentage of top questions to collect (default 10%)

        Yields:
            Tuple of (question_data_dict, api_quota_remaining)
        """
        max_creation_date = int((datetime.now() - timedelta(days=min_age_days)).timestamp())
        page = 1
        has_more = True

        logger.info(f"[{site}] Searching for top {top_percentage}% questions with "
                    f"score >= {min_score=}, age >= {min_age_days=} days")

        # Phase 1: Determine total unanswered question pool
        try:
            total_no_answer_params = {
                'page': 1,
                'pagesize': 1,
                'site': site,
                'filter': 'total',  # Only get total count
                'answers': 0,  # Only questions with no answers
            }
            no_answer_data = self._make_request("questions/no-answers", total_no_answer_params)
            num_Qs_no_answer = no_answer_data.get("total", 0)
            if num_Qs_no_answer == 0:
                logger.warning(f"[{site}] No no-answer questions found on the site.")
                return

            # Calculate how many questions represent the top X%
            num_top_Qs = max(1, int(num_Qs_no_answer * (top_percentage / 100.0)))
            logger.info(f"[{site}] Found {num_Qs_no_answer} no-answer questions. "
                        f"Target: top {num_top_Qs} questions ({top_percentage}%)")

            # Phase 2: Count questions meeting our quality criteria
            total_filtered_params = {
                'page': 1,
                'pagesize': 100,
                'order': 'desc',
                'sort': 'votes',
                'site': site,
                'filter': 'total',
                'todate': max_creation_date,
                'min': min_score,
                'answers': 0,
            }
            filtered_data = self._make_request("questions/no-answers", total_filtered_params)
            # NOTE: Stack Exchange API 'total' counts can be inaccurate
            # Always verify against actual site UI if precision is critical
            num_eligible_Qs = filtered_data.get("total", 0)
            if num_eligible_Qs == 0:
                logger.info(f"[{site}] No eligible questions found matching criteria.")
                return

            logger.info(f"[{site}] {num_eligible_Qs} questions match filtering criteria")
            collected_count = 0

        except Exception as e:
            logger.warning(f"[{site}] Failed to determine question counts: {e}")
            logger.warning("This may indicate API quota exhaustion or request issues")
            num_top_Qs = collected_count = num_eligible_Qs = 0
            has_more = False

        # Phase 3: Collect questions in vote-ranked order
        logger.info(f"[{site}] Beginning question collection ({collected_count}/{num_eligible_Qs})")
        while has_more and collected_count < num_eligible_Qs:
            params = {
                'page': page,
                'pagesize': page_size,
                'order': 'desc',
                'sort': 'votes',  # Get highest-voted questions first
                'site': site,
                'filter': 'withbody',  # Include question body for preliminary filtering
                'todate': max_creation_date,
                'min': min_score,
                'answers': 0,  # Only questions with no answers
            }

            try:
                data = self._make_request("questions/no-answers", params)
                # Extract questions and monitor API quota
                question_data_list = data.get("items", [])
                quota_remaining = data.get("quota_remaining", ASSUMED_QUOTA_REMAINING)
                if not question_data_list:
                    logger.info(f"[{site}] No items found on page {page}. Stopping.")
                    break

                # Yield questions up to our target % limit
                for question_data in question_data_list:
                    if collected_count < num_eligible_Qs:
                        yield question_data, quota_remaining
                        collected_count += 1
                    else:
                        logger.info(f"[{site}] Reached target of {num_eligible_Qs} questions "
                                    f"({top_percentage}% of unanswered). Stopping collection.")
                        has_more = False
                        break

                if has_more:
                    has_more = data.get("has_more", False)
                    page += 1
                    if quota_remaining <= 10:
                        logger.warning(f"[{site}] Low API quota remaining: {quota_remaining}")

                logger.debug(f"[{site}] {collected_count=}, {num_eligible_Qs=}, {has_more=}")

            except Exception as e:
                logger.warning(f"[{site}] API request failed on page {page}: {e}")
                logger.info("Waiting before retry...")
                time.sleep(MIN_RETRY_DELAY)
                continue

        logger.info(f"[{site}] Question crawling complete. Collected {collected_count} candidates.")

    def get_comments(self, question_id: str, site: str) -> Tuple[List[str], int]:
        """Retrieve all comments for a specific question.

        Args:
            question_id: Stack Exchange question ID
            site: Stack Exchange site name

        Returns:
            Tuple of (comment_texts_list, api_quota_remaining)
        """
        comments_endpoint = f"questions/{question_id}/comments"
        comments_params = {"site": site, "sort": "creation", "filter": "withbody"}
        comments_data = self._make_request(comments_endpoint, comments_params)
        quota_remaining = comments_data.get("quota_remaining")
        comments = [html_to_text(comment['body']) for comment in comments_data.get('items', [])]
        return comments, quota_remaining

    def get_question_details(self, question_id: str, site: str) -> Tuple[dict, Optional[int]]:
        """Retrieve details for a specific question.

        This method fetches complete question information including:
        - Question content (title, body, tags)
        - Metadata (score, views, creation date)
        - All associated comments

        Args:
            question_id: Stack Exchange question ID
            site: Stack Exchange site name

        Returns:
            Tuple of (formatted_question_data_dict, api_quota_remaining)
            Returns empty dict if question not found or API error occurs
        """
        human_url = f"https://{site}.stackexchange.com/questions/{question_id}"
        quota = None  # Initialize quota
        try:
            # Fetch core question details
            question_endpoint = f"questions/{question_id}"
            question_params = {
                "site": site,
                "filter": "withbody",  # Include the question body
                "order": "desc",
                "sort": "activity",
                "include": "body_markdown,body,tags,score,creation_date,link,title,comment_count"
            }
            question_data = self._make_request(question_endpoint, question_params)
            quota_remaining = question_data.get("quota_remaining", ASSUMED_QUOTA_REMAINING)

            # Check if we have sufficient quota for additional requests
            if quota_remaining <= 4:
                logger.warning(f"[{site}] Insufficient API quota ({quota_remaining}). "
                             "Skipping detailed processing.")
                return {}, quota_remaining

            if not question_data or not question_data.get("items"):
                logger.warning(f"[{site}] No details found for question ID {question_id}")
                return {}, quota_remaining

            question = question_data["items"][0]

            # Fetch associated comments
            comments, quota_remaining = self.get_comments(question_id, site)

            # Format standardized question data
            creation_date = datetime.fromtimestamp(question.get("creation_date", 0)).isoformat()
            # Construct standardized question data structure
            final_data = {
                "id":
                    str(question["question_id"]),
                "site":
                    site,
                "link":
                    question.get("link", human_url),
                "title":
                    question.get("title", ""),
                "body":
                    html_to_text(question.get("body", "")),
                "body_markdown":
                    question.get("body_markdown", html_to_markdown(question.get("body", ""))),
                "answer":
                    question.get("answer", "[No answer]"),
                "tags":
                    question.get("tags", []),
                "score":
                    question.get("score", -1),
                "views":
                    question.get("view_count", -1),
                "creation_date":
                    creation_date,
                "comments":
                    comments,
                "comment_count":
                    len(comments),
            }

            logger.debug(f"[{site}] Successfully retrieved complete details for question {question_id}")
            return final_data, quota_remaining

        except Exception as e:
            logger.error(f"[{site}] Failed to fetch details for question {question_id}: {e}")
            return {}, quota_remaining
