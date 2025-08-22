import os
import argparse
import time
import logging
import signal
import sys
import json
from datetime import datetime, timedelta
from threading import Lock
from loguru import logger
from dotenv import load_dotenv
from stackexchange_client import StackExchangeClient
from filters import filter_check
from diamond_config import get_diamond_settings

# --- Load Stack Exchange API key (strongly recommended) ---
# Register at https://api.stackexchange.com/ to get your free API key
# Place it in a key.env file as: STACKEXCHANGE_API_KEY=your_key_here
load_dotenv("key.env")
load_dotenv("../key.env")
STACKEXCHANGE_API_KEY = os.getenv("STACKEXCHANGE_API_KEY")

# --- Constants ---
SAVE_DIR = "outputs"
PREFIX = "__crawler"

# Site configurations - see sites.jsonl for complete list of available sites
# Use the `api_site_parameter` field from https://api.stackexchange.com/docs/sites
DEFAULT_SITE = "biology"
DEFAULT_SITES = ["math", "physics", "mathoverflow.net", "chemistry", "stats", "cstheory"]
DEFAULT_OUTPUT_FILE = f"{PREFIX}_questions.jsonl"
DEFAULT_STATE_FILE = f"{PREFIX}_processed_qids.txt"

# Question filtering thresholds
DEFAULT_MIN_SCORE = 5  # Minimum upvotes for a question to be considered
DEFAULT_MIN_VIEWS = 500  # Minimum views for question engagement
DEFAULT_MIN_AGE_DAYS = 365 * 2  # Questions must be at least 2 years old
DEFAULT_TOP_PERCENTAGE = 10.0  # Collect top 10% of highest voted questions

# Crawling parameters
DEFAULT_API_DELAY_SECONDS = 0.05  # Conservative delay between API requests
SAVE_STATE_INTERVAL = 10  # Save progress every N questions processed

# --- Logging Setup ---
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Log INFO and above to stderr


# --- Signal handling for graceful shutdown ---
def handle_shutdown(signum, frame):
    """Handle shutdown signals (Ctrl+C, etc.) gracefully."""
    logger.warning(f"Shutdown signal ({signum}) received. Exiting immediately...")
    # State is saved atomically after each question, so no cleanup needed
    sys.exit(0)


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGQUIT, handle_shutdown)


# --- Helper Classes ---
class OutputWriter:
    """Handles thread-safe writing of valid questions to JSONL output file."""

    def __init__(self, output_file):
        self.output_file = output_file
        self.lock = Lock()
        logging.info(f"OutputWriter initialized for file {output_file}")

    def write_question(self, question_data):
        # Ensure data is serializable
        def default_serializer(o):
            if isinstance(o, (datetime, timedelta)):
                return o.isoformat()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        try:
            line = json.dumps(question_data, default=default_serializer)
            with self.lock:
                # Open and close the file for each write to prevent data loss on unexpected termination
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
                    f.flush()  # Ensure it's written to disk
            logging.info(f"Appended question {question_data['id']} to {self.output_file}")
        except Exception as e:
            logging.error(
                f"Failed to write question {question_data.get('id', 'UNKNOWN')} to {self.output_file}: {e}"
            )


class StateManager:
    """Manages persistent state of processed question IDs to enable resuming interrupted crawls."""

    def __init__(self, state_file):
        self.state_file = state_file
        self.lock = Lock()
        self.processed_ids = self._load_processed_ids()
        logger.info(f"Loaded {len(self.processed_ids)} processed question IDs from {state_file}")

    def _load_processed_ids(self):
        with self.lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r') as f:
                        # Read IDs line by line to handle potentially large files
                        return set(line.strip() for line in f if line.strip())
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(
                        f"Error loading state file {self.state_file}: {e}. Starting fresh.")
                    return set()
            else:
                return set()

    def is_processed(self, question_id):
        return str(question_id) in self.processed_ids

    def add_processed(self, question_id):
        question_id_str = str(question_id)
        with self.lock:
            # Add to in-memory set
            self.processed_ids.add(question_id_str)
            # Write directly to file
            try:
                with open(self.state_file, 'a') as f:
                    f.write(f"{question_id_str}\n")
                    f.flush()  # Ensure it's written to disk
            except IOError as e:
                logger.error(f"Error writing to state file {self.state_file}: {e}")

    def save_state(self):
        # This method is kept for interface compatibility
        # Since we're writing each ID as it's processed, we don't need to do anything here
        logger.info(f"State is already saved with {len(self.processed_ids)} processed IDs")


# --- Command Line Interface ---
def commandline_args():
    parser = argparse.ArgumentParser(description="Stack Exchange Unanswered Question crawler")
    parser.add_argument(
        "--sites",
        nargs='+',
        default=DEFAULT_SITES,
        help=f"(Deprecated) List of sites to crawl. Use --site instead. See sites.jsonl for available sites. Default: {DEFAULT_SITES}")
    parser.add_argument("--site",
                        default=DEFAULT_SITE,
                        required=True,
                        help=f"Stack Exchange site to crawl (e.g., 'math', 'physics'). See sites.jsonl for all available sites. Default: {DEFAULT_SITE}")
    parser.add_argument(
        "--output_file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"File to save collected questions (JSON Lines format). Default: {DEFAULT_OUTPUT_FILE}"
    )
    parser.add_argument(
        "--state_file",
        default=DEFAULT_STATE_FILE,
        help=f"File to store IDs of processed questions. Default: {DEFAULT_STATE_FILE}")
    parser.add_argument("--tag", help=f"Tag for output files.")
    parser.add_argument("--min_score",
                        type=int,
                        default=DEFAULT_MIN_SCORE,
                        help=f"Minimum score for questions. Default: {DEFAULT_MIN_SCORE}")
    parser.add_argument("--min_views",
                        type=int,
                        default=DEFAULT_MIN_VIEWS,
                        help=f"Minimum views for questions. Default: {DEFAULT_MIN_VIEWS}")
    parser.add_argument("--min_age_days",
                        type=int,
                        default=DEFAULT_MIN_AGE_DAYS,
                        help=f"Minimum age of questions in days. Default: {DEFAULT_MIN_AGE_DAYS}")
    parser.add_argument(
        "--api_key",
        default=STACKEXCHANGE_API_KEY,
        help="(Deprecated) Stack Exchange API Key - set this in key.env file instead. Increases API rate limit from 300 to 10,000 requests per day."
    )
    parser.add_argument(
        "--api_delay",
        type=float,
        default=DEFAULT_API_DELAY_SECONDS,
        help=f"Delay in seconds between fetching each question. Default: {DEFAULT_API_DELAY_SECONDS}"
    )
    parser.add_argument(
        "--top_percentage",
        type=float,
        default=DEFAULT_TOP_PERCENTAGE,
        help=
        f"Collect only the top X% of highest voted questions. Default: {DEFAULT_TOP_PERCENTAGE}%")

    # --- Diamond Mode: High-Quality Subset ---
    parser.add_argument("--diamond", action="store_true",
                        help="Enable diamond mode for high-quality question subset with stricter filtering criteria")

    args = parser.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.diamond:
        # Use site-specific diamond settings (site is always provided)
        diamond_settings = get_diamond_settings(args.site)
        args.min_score = diamond_settings["min_score"]
        args.min_views = diamond_settings["min_views"]
        args.min_age_days = diamond_settings["min_age_days"]
        args.top_percentage = diamond_settings["top_percentage"]
        logger.info(f"---------- Diamond Subset for {args.site} ----------\n"
                    f"Using stricter criteria: min_score={args.min_score}, "
                    f"min_views={args.min_views}, min_age_days={args.min_age_days}, "
                    f"top_percentage={args.top_percentage}%")
        args.tag = "diamond"

    prefix = f"{PREFIX}_{args.tag}" if args.tag else PREFIX

    if args.site:
        logger.info(f"Single-site mode: {args.site}")
        args.sites = [args.site]

        # Configure output files with site-specific names
        args.output_file = f"{SAVE_DIR}/{prefix}_{args.site}_questions.jsonl"
        args.state_file = f"{SAVE_DIR}/{prefix}_{args.site}_processed_qids.txt"
        logger.info(f"Output file: {args.output_file}")
        logger.info(f" State file: {args.state_file}")

        # Set up detailed logging to file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.log_file = f"{SAVE_DIR}/{prefix}_{args.site}_{timestamp}.log"
        logger.add(args.log_file, level="DEBUG", rotation="10 MB")

    # Record the command line invocation for reproducibility
    logger.info(f"Command executed: {' '.join(sys.argv)}")
    return args


# --- Main Function ---
def main(args):
    """Main function to orchestrate the question collection process."""
    logger.info("Starting question crawler...")
    logger.info(f"Configuration: {args}")

    # Initialize components
    client = StackExchangeClient(api_key=args.api_key)
    state_manager = StateManager(args.state_file)
    output_writer = OutputWriter(args.output_file)

    try:
        for site in args.sites:
            logger.info(f"[{site}] Searching for candidate questions...")
            question_ids_generator = client.find_potential_questions(
                site=site,
                min_score=args.min_score,
                min_age_days=args.min_age_days,
                page_size=100,
                top_percentage=args.top_percentage)

            for (question_data, _) in question_ids_generator:
                question_id = str(question_data['question_id'])
                if state_manager.is_processed(question_id):
                    logger.info(f"[{site}] QID {question_id}: Already processed, skipping.")
                    continue

                # Introduce delay *before* fetching details to respect rate limits
                logger.debug(f"[{site}] Processing QID {question_id}... {args.api_delay=}")
                time.sleep(args.api_delay)

                # Fetch complete question details from API
                question_data, quota = client.get_question_details(question_id, site)

                if not question_data:
                    logger.debug(f"[{site}] QID {question_id}: Empty question data, skipping.")
                    state_manager.add_processed(question_id)
                    continue  #

                # --- perform rule-based filters ---
                filter_result, reason = filter_check(question_data, site, args.min_score, args.min_age_days, args.min_views)
                if filter_result:
                    output_writer.write_question(question_data)
                    title = question_data.get('title', 'No title')
                    logger.info(f"[{site}] QID {question_id}: Passed filters. Saved. "
                                f"API Quota: {quota}. API delay: {args.api_delay}. "
                                f"Title: '{title}'")
                else:
                    logger.info(f"[{site}] QID {question_id}: Failed filters ({reason}). Skipping.")

                # Mark as processed regardless of whether it passed filters
                state_manager.add_processed(question_id)

            logger.info(f"[{site}] Finished processing candidates. Moving to next site.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")  # Log stack trace
    finally:
        # State is automatically saved after each processed question
        logger.info(f"Crawler finished. Results saved to: {args.output_file}")
        if hasattr(args, 'log_file'):
            logger.info(f"Detailed logs available at: {args.log_file}")


# --- entry point ---
if __name__ == "__main__":
    args = commandline_args()
    main(args)
