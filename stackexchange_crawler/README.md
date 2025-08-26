# Stack Exchange Question Crawler + Rule-Based Filters

A Python toolkit for crawling unanswered questions from Stack Exchange sites. This crawler is designed to collect questions that remain unanswered despite community attention (see paper section 2.1).

## Features

* **Question discovery**: Finds unanswered questions ranked by community engagement (votes, views)
* **Filtering rules**: Defines the rules used for filtering (age, views, votes, ...)
* **Site-specific rules**: Customized filtering criteria for different Stack Exchange communities
* **Resumable crawling**: State management allows interrupted crawls to be resumed (i.e. resume from Ctrl + C)
* **Rate limiting**: Respects Stack Exchange API guidelines with configurable delays
* **Logging**: Detailed logs for monitoring and debugging
* **Diamond subset**: Stricter filtering

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install requests beautifulsoup4 html2text loguru python-dotenv
```

### API Key Setup (recommended but optional; increases rate limit)

1. Register for a free Stack Exchange API key at https://api.stackexchange.com/
2. Create a `key.env` file in the project directory:

```
STACKEXCHANGE_API_KEY=your_api_key_here
```

Using an API key increases your rate limit from 300 to 10, 000 requests per day.

### Basic Usage

```bash
# Crawl unanswered questions from a specific site (interrupt at any time)
python dataset_crawler.py --site math --min_score 10 --min_views 500

# Crawl diamond subset candidates
python dataset_crawler.py --site physics --diamond

# Crawl a single question by URL
python oneoff_crawler.py https://math.stackexchange.com/questions/358423/a-proof-of-dimrt-dimr1-without-prime-ideals
```

### Resume Capability

Interrupted crawls can be resumed by simply running the same command:

```bash
# This will skip already processed questions
python dataset_crawler.py --site math --min_score 10 --min_views 500
```

## Main Components

### 1. Dataset Crawler ( `dataset_crawler.py` )

The primary tool for systematic question collection from Stack Exchange sites.

#### Usage

```bash
python dataset_crawler.py --site <SITE_NAME> [OPTIONS]
```

#### Key Arguments

* `--site`: Stack Exchange site to crawl (required). See `sites.jsonl` for available sites
* `--min_score`: Minimum upvotes required (default: 5)
* `--min_views`: Minimum view count (default: 500)
* `--min_age_days`: Minimum question age in days (default: 730)
* `--top_percentage`: Percentage of top questions to collect (default: 10%)
* `--api_delay`: Delay between API requests in seconds (default: 0.05)
* `--diamond`: Enable diamond mode for strictest quality filtering

#### Examples

```bash
# Basic crawling with default settings
python dataset_crawler.py --site biology

# High-quality subset with custom thresholds
python dataset_crawler.py --site math --min_score 25 --min_views 1000 --top_percentage 5

# Diamond subset (uses predefined strict criteria)
python dataset_crawler.py --site physics --diamond

# Batch processing multiple sites
bash diamond.sh
```

#### Outputs and interruption handling

The crawler maintains state to enable resuming interrupted crawls:
* **Atomic Updates**: State is saved after each processed question
* **Graceful Shutdown**: Ctrl+C safely stops crawling without data loss

All outputs are saved in the `outputs/` directory:
* `__crawler_{site}_questions.jsonl`: Collected questions in JSON Lines format
* `__crawler_{site}_processed_qids.txt`: IDs of processed questions (for resuming)
* `__crawler_{site}_{timestamp}.log`: Detailed crawling logs

### 2. One-off Crawler ( `oneoff_crawler.py` )

Tool for crawling individual questions by URL:

```bash
python oneoff_crawler.py "https://math.stackexchange.com/questions/893875/sorting-of-prime-gaps"
```

This outputs both structured JSON metadata and formatted text content.

## Rule-Based Filters

The crawler implements multiple rule-based filters (discussed in paper section 2.1):

**Engagement metrics**:
* **Minimum Score**: Questions must have sufficient upvotes
* **Minimum Views**: Questions must show community interest
* **View-to-Vote Ratio**: Filters out questions with many views but few votes
* **Minimum Age**: Ensures questions have had time to attract answers

**Content quality**:
* **Image Filtering**: Excludes questions containing images (focuses on text-based problems)
* **Tag Filtering**: Removes recommendation requests, homework, opinion-based questions
* **"Why" Questions**: Filters subjective "why" questions from titles
* **Site-Specific Rules**: Custom filtering for different academic domains

**Site-specific filtering** (different sites may have tailored filtering criteria):
* **Mathematics sites** (`math`,   `mathoverflow.net`): Higher score thresholds, filters homework
* **Programming sites** (`stackoverflow`,   `cstheory`): Removes career advice, "best practice" questions
* **Physics site**: Filters homework and recommendation requests
* **See `filters.py`** for complete filtering logic

## Available Sites

The crawler supports all Stack Exchange sites. Common sites include:

* `math` - Mathematics
* `mathoverflow.net` - MathOverflow (research mathematics)
* `physics` - Physics
* `chemistry` - Chemistry
* `biology` - Biology
* `stats` - Cross Validated (Statistics)
* `cstheory` - Theoretical Computer Science
* `stackoverflow` - Stack Overflow (Programming)
* `tex` - TeX/LaTeX

See `sites.jsonl` for the complete list of 300+ available sites.

## Configuration

### API Rate Limiting

The crawler includes intelligent rate limiting:

```bash
# Conservative rate limiting (recommended)
python dataset_crawler.py --site math --api_delay 0.1

# Faster crawling (use with caution)
python dataset_crawler.py --site math --api_delay 0.02
```

### Modifications

1. **Filtering Logic**: Update `filters.py` for new quality criteria

   * Add new filtered tags to `EXACT_FILTERED_TAGS`
   * Customize site-specific rules in the site validation functions
   * Adjust quality thresholds like `MAX_VIEW_TO_VOTE_RATIO`
   * Edit `diamond_config.py` if needed

2. **API Changes**: Modify `stackexchange_client.py` for API updates

## Output Format

Each collected question is saved as a JSON object with the following fields:

```json
{
  "id": "123456",
  "site": "math",
  "link": "https://math.stackexchange.com/questions/123456",
  "title": "Question title",
  "body": "Plain text question body",
  "body_markdown": "Markdown formatted body",
  "answer": "[No answer]",
  "tags": ["algebra", "number-theory"],
  "score": 25,
  "views": 1500,
  "creation_date": "2021-03-15T10:30:00",
  "comments": ["Comment 1", "Comment 2"],
  "comment_count": 2
}
```

## Performance Considerations

* **Without API key**: 300 requests/day per IP
* **With API key**: 10, 000 requests/day
* **Question details**: ~2 requests per question (question + comments)
* **Expected throughput**: 100-5000 questions/day depending on API key

## Troubleshooting

Common issues:

1. **API Quota Exceeded**: get an API key or wait for couple hours.
2. **No Questions Found**: check if site name is correct (see sites.jsonl) or lower quality thresholds (min_score, min_views)
3. **Rate Limiting Errors**: increase --api_delay parameter

Detailed logs are available in `outputs/__crawler_{site}_{timestamp}.log` :

## License

CC BY-SA 4.0, following https://stackoverflow.com/legal/terms-of-service/public#licensing.

## Citation

If you find this crawler helpful, please consider citing the UQ project! See main repo README for BibTeX.
