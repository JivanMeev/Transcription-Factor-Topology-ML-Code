#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_FILE="$DATA_DIR/uniprot_data.tab"

STREAM_GZ="$DATA_DIR/uniprot_data.stream.tab.gz"
STREAM_TSV="$DATA_DIR/uniprot_data.stream.tab"

# Exact original query and exact original requested fields.
QUERY='reviewed:true AND length:[50 TO 5500]'
FIELDS='accession,sequence,go,organism_name,length,protein_existence'

# Exact URL used by the original repository script.
STREAM_URL='https://rest.uniprot.org/uniprotkb/stream?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%205500%5D&format=tsv&fields=accession,sequence,go,organism_name,length,protein_existence&compressed=true'

mkdir -p "$DATA_DIR"

# Remove only temporary files. An existing valid dataset is not deleted
# unless a new download succeeds.
rm -f "$STREAM_GZ" "$STREAM_TSV"

printf '\nAttempting UniProt download with /stream first...\n\n'

stream_ok=false

# ------------------------------------------------------------
# Attempt 1: original /stream method
# ------------------------------------------------------------

if curl \
    --fail \
    --location \
    --retry 3 \
    --retry-delay 3 \
    --connect-timeout 30 \
    --output "$STREAM_GZ" \
    "$STREAM_URL"
then
    # Confirm it is valid gzip data, then decompress it temporarily.
    if gzip -t "$STREAM_GZ" 2>/dev/null &&
       gunzip -c "$STREAM_GZ" > "$STREAM_TSV"
    then
        line_count="$(wc -l < "$STREAM_TSV" | tr -d ' ')"

        # The stream endpoint can return HTTP 200 and valid gzip while
        # embedding an error message inside the resulting TSV. Therefore,
        # validate the content rather than trusting curl's exit status alone.
        if head -n 1 "$STREAM_TSV" | grep -q $'^Entry\tSequence' &&
           ! grep -q 'Error encountered when streaming data' "$STREAM_TSV" &&
           [ "$line_count" -gt 1000 ]
        then
            stream_ok=true
        fi
    fi
fi

if [ "$stream_ok" = true ]; then
    mv "$STREAM_TSV" "$OUTPUT_FILE"
    rm -f "$STREAM_GZ"

    printf '\n/stream succeeded.\n'
    printf 'Saved dataset to: %s\n' "$OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"

    exit 0
fi

# ------------------------------------------------------------
# Stream failed: explain what happened
# ------------------------------------------------------------

printf '\nWARNING: UniProt /stream did not return a complete dataset.\n' >&2

if [ -f "$STREAM_TSV" ]; then
    printf 'The beginning of the returned response was:\n\n' >&2
    head -n 5 "$STREAM_TSV" >&2
fi

rm -f "$STREAM_GZ" "$STREAM_TSV"

printf '\nSwitching automatically to paginated UniProt /search...\n'
printf 'The query and requested fields are unchanged.\n\n'

# ------------------------------------------------------------
# Attempt 2: paginated /search fallback
# ------------------------------------------------------------

python3 - "$OUTPUT_FILE" "$QUERY" "$FIELDS" <<'PY'
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


output_file, query, fields = sys.argv[1:]
temporary_file = output_file + ".search.part"

parameters = {
    "query": query,
    "format": "tsv",
    "fields": fields,
    "size": "500",
}

next_url = (
    "https://rest.uniprot.org/uniprotkb/search?"
    + urllib.parse.urlencode(parameters)
)

expected_header = None
downloaded_entries = 0
total_entries = None
page_number = 0
start_time = time.monotonic()


def format_duration(seconds):
    """Convert seconds into MM:SS or HH:MM:SS."""

    if seconds is None or seconds < 0:
        return "--:--"

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def show_progress(done, total):
    """Display one continuous progress bar for all search pages."""

    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = done / elapsed

    if total and total > 0:
        fraction = min(done / total, 1.0)
        percent = fraction * 100

        width = 38
        completed = int(width * fraction)

        if completed >= width:
            bar = "=" * width
        else:
            bar = (
                "=" * completed
                + ">"
                + " " * (width - completed - 1)
            )

        eta = (total - done) / rate if rate > 0 else None

        message = (
            f"\r{percent:6.2f}% [{bar}] "
            f"{done:,}/{total:,} entries  "
            f"{rate:,.0f} entries/s  "
            f"ETA {format_duration(eta)}"
        )
    else:
        # Fallback display if UniProt omits X-Total-Results.
        message = (
            f"\rDownloaded {done:,} entries "
            f"from {page_number:,} pages  "
            f"{rate:,.0f} entries/s"
        )

    sys.stdout.write(message)
    sys.stdout.flush()


def download_page(url, maximum_attempts=6):
    """Download one page, retrying temporary failures."""

    for attempt in range(1, maximum_attempts + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/tab-separated-values",
                "User-Agent": (
                    "Transcription-Factor-Topology-ML-Code/1.0"
                ),
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=180,
            ) as response:
                return response.read(), response.headers

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
        ) as error:
            if attempt == maximum_attempts:
                raise RuntimeError(
                    "UniProt /search failed after "
                    f"{maximum_attempts} attempts: {error}"
                ) from error

            retry_after = None

            if isinstance(error, urllib.error.HTTPError):
                retry_after_header = error.headers.get(
                    "Retry-After"
                )

                if (
                    retry_after_header
                    and retry_after_header.isdigit()
                ):
                    retry_after = int(retry_after_header)

            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(60, 2 ** attempt)

            # Move below the active progress-bar line.
            sys.stdout.write("\n")

            print(
                f"Page request failed: {error}. "
                f"Retrying in {delay} seconds "
                f"({attempt}/{maximum_attempts})...",
                file=sys.stderr,
            )

            time.sleep(delay)


try:
    with open(temporary_file, "wb") as output:
        while next_url:
            page_number += 1

            body, headers = download_page(next_url)

            if b"Error encountered when streaming data" in body:
                raise RuntimeError(
                    "UniProt returned an error message "
                    "instead of TSV data."
                )

            lines = body.splitlines(keepends=True)

            if not lines:
                raise RuntimeError(
                    f"UniProt /search page {page_number} "
                    "was empty."
                )

            current_header = lines[0].rstrip(b"\r\n")

            if expected_header is None:
                if not current_header.startswith(
                    b"Entry\tSequence"
                ):
                    preview = body[:500].decode(
                        "utf-8",
                        errors="replace",
                    )

                    raise RuntimeError(
                        "UniProt did not return the expected "
                        f"TSV header:\n{preview}"
                    )

                expected_header = current_header

                # Keep the header from the first page.
                output.writelines(lines)
            else:
                if current_header != expected_header:
                    raise RuntimeError(
                        f"Page {page_number} returned a "
                        "different TSV header."
                    )

                # Later pages repeat the header, so skip it.
                output.writelines(lines[1:])

            page_entries = max(0, len(lines) - 1)
            downloaded_entries += page_entries

            if total_entries is None:
                total_header = headers.get(
                    "X-Total-Results"
                )

                if total_header and total_header.isdigit():
                    total_entries = int(total_header)

            show_progress(
                downloaded_entries,
                total_entries,
            )

            # UniProt provides the next cursor URL in the
            # HTTP Link header.
            link_header = headers.get("Link", "")

            next_match = re.search(
                r'<([^>]+)>\s*;\s*rel="next"',
                link_header,
            )

            if next_match:
                next_url = next_match.group(1)
            else:
                next_url = None

    sys.stdout.write("\n")

    # This original query should produce far more than 1,000
    # proteins, so a tiny response indicates another failure.
    if downloaded_entries < 1000:
        raise RuntimeError(
            f"Only {downloaded_entries} entries were "
            "downloaded; the result is suspiciously small."
        )

    # Replace the old dataset only after the complete fallback
    # download succeeds.
    os.replace(temporary_file, output_file)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    elapsed = time.monotonic() - start_time

    print("\n/search download succeeded.")
    print(f"Entries downloaded: {downloaded_entries:,}")
    print(f"Elapsed time: {format_duration(elapsed)}")
    print(f"Output file: {output_file}")
    print(f"Output size: {size_mb:.1f} MB")

except Exception:
    if os.path.exists(temporary_file):
        os.remove(temporary_file)

    raise
PY
