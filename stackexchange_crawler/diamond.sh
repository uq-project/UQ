#!/bin/bash

# Diamond subset batch crawler.
#
# Add or remove sites as needed. Site names correspond to the api_site_parameter
# field in sites.jsonl. See https://api.stackexchange.com/docs/sites for reference.

SITES="cstheory math mathoverflow.net physics puzzling scifi stackoverflow.com tex"

echo "Starting diamond mode crawling for sites: $SITES"
echo "This may take a while depending on API rate limits..."

for site in $SITES; do
    echo "=== Processing site: $site ==="
    python dataset_crawler.py --site $site --api_delay 0.05 --diamond
    echo "=== Completed site: $site ==="
    echo
done

echo "Diamond subset crawling completed for all sites."

