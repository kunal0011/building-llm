"""
Distributed Common Crawl WARC Ingestion and Trafilatura Extraction Engine.
Streams compressed WARC records directly from S3 or HTTPS, applies HTML extraction,
heuristic filtering, and extracts multimodal anchors.
"""

import io
import re
import gzip
import urllib.request
from typing import Iterator, Dict, Any, Optional, List
import warcio
from warcio.archiveiterator import ArchiveIterator
import trafilatura

from schemas.data_schema import FilteredDocument, MediaReference
from pipelines.filtering.heuristic_filter import HeuristicQualityFilter


class CommonCrawlWarcExtractor:
    """
    High-throughput WARC stream extractor for web pre-training data.
    Supports both direct S3 reading and HTTPS streaming.
    """

    COMMON_CRAWL_BASE_URL = "https://data.commoncrawl.org/"

    def __init__(
        self,
        heuristic_filter: Optional[HeuristicQualityFilter] = None,
        extract_images: bool = True,
        min_doc_length: int = 150
    ):
        self.heuristic_filter = heuristic_filter or HeuristicQualityFilter()
        self.extract_images = extract_images
        self.min_doc_length = min_doc_length

        # Regex for fast img extraction
        self.img_regex = re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*alt=["\']([^"\']*)["\']', re.IGNORECASE)

    def open_warc_stream(self, warc_path_or_url: str) -> io.BytesIO:
        """
        Opens a WARC stream from a local file, S3 path, or Common Crawl relative URL.
        Example relative URL: 'crawl-data/CC-MAIN-2024-10/segments/.../warc/....warc.gz'
        """
        if warc_path_or_url.startswith("http://") or warc_path_or_url.startswith("https://"):
            url = warc_path_or_url
        elif warc_path_or_url.startswith("crawl-data/"):
            url = f"{self.COMMON_CRAWL_BASE_URL}{warc_path_or_url}"
        else:
            # Local file path
            return open(warc_path_or_url, "rb")

        # Stream via HTTP with custom User-Agent
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AITrainingDataCrawler/1.0 (Foundation Model Research; +https://example.org/bot)"}
        )
        response = urllib.request.urlopen(req)
        return io.BytesIO(response.read())

    def extract_records_from_stream(self, stream: io.IOBase) -> Iterator[Dict[str, Any]]:
        """
        Iterates over WARC response records in a byte stream.
        """
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            content_type = record.http_headers.get_header("Content-Type", "") if record.http_headers else ""
            if not ("text/html" in content_type.lower() or "application/xhtml+xml" in content_type.lower()):
                continue

            status_code = record.http_headers.get_statuscode() if record.http_headers else ""
            status_line = getattr(record.http_headers, "statusline", "") if record.http_headers else ""
            protocol = getattr(record.http_headers, "protocol", "") if record.http_headers else ""
            if status_code != "200" and protocol != "200" and "200" not in status_line:
                continue

            url = record.rec_headers.get_header("WARC-Target-URI", "")
            raw_payload = record.content_stream().read()

            try:
                html_str = raw_payload.decode("utf-8", errors="replace")
            except Exception:
                continue

            # Extract media references (images with alt-texts) for multimodal alignment
            media_refs: List[MediaReference] = []
            if self.extract_images:
                matches = self.img_regex.findall(html_str)[:10]  # Cap at 10 images per doc
                for idx, (src, alt) in enumerate(matches):
                    if alt.strip():
                        media_refs.append(MediaReference(
                            placeholder_token=f"<|image_ref_{idx:03d}|>",
                            s3_uri=src,
                            alt_text=alt.strip(),
                            mime_type="image/jpeg"
                        ))

            # Main text extraction via Trafilatura
            extracted_text = trafilatura.extract(
                html_str,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )

            if not extracted_text or len(extracted_text) < self.min_doc_length:
                continue

            # Apply heuristic and repetition filtering
            passed, reason, metrics = self.heuristic_filter.evaluate(extracted_text)
            if not passed:
                continue

            # Scrub PII
            clean_text = self.heuristic_filter.scrub_pii(extracted_text)

            yield {
                "url": url,
                "text": clean_text,
                "metrics": metrics,
                "media_refs": [ref.__dict__ for ref in media_refs]
            }

    def process_warc_file(self, warc_source: str) -> List[Dict[str, Any]]:
        """Extracts and filters all valid documents from a single WARC source."""
        stream = self.open_warc_stream(warc_source)
        try:
            return list(self.extract_records_from_stream(stream))
        finally:
            if hasattr(stream, "close"):
                stream.close()
