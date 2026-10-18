"""
Image Quality and Dimension Validator for Multimodal Pretraining.
Filters out tracking pixels, ad banners, icons, placeholder graphics,
and uninformative boilerplate alt-texts prior to vision-language alignment.
"""

import re
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any


class ImageFilterValidator:
    """
    Validates image URLs, dimensions, aspect ratios, and alt-text semantics.
    Enforces quality standards for frontier multimodal foundation models.
    """

    ALLOWED_MIME_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/svg+xml"
    }

    ALLOWED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"
    }

    GENERIC_ALT_PATTERNS = [
        r"^(image|photo|picture|graphic|logo|icon|avatar|banner|thumbnail|spacer)$",
        r"^(untitled|default|placeholder|no\s*description|n/?a|none)$",
        r"^(screenshot|scan|img|dsc|pic|figure)[_\-\s]*\d*$",
        r"^[a-zA-Z0-9_\-]+\.(jpg|jpeg|png|gif|webp|svg)$",  # Just a filename
        r"^\s*$",  # Empty or whitespace only
        r"^\d+\s*(x|×)\s*\d+$"  # Dimension strings like '800x600'
    ]

    DISALLOWED_URL_PATTERNS = [
        r"/tracking/",
        r"/pixel\.",
        r"/analytics",
        r"/beacon",
        r"/ad[s]?/",
        r"/adserver",
        r"1x1",
        r"badge",
        r"favicon"
    ]

    def __init__(
        self,
        min_width: int = 120,
        min_height: int = 120,
        max_width: int = 8192,
        max_height: int = 8192,
        min_pixel_area: int = 25_000,
        min_aspect_ratio: float = 0.20,
        max_aspect_ratio: float = 5.0,
        min_alt_words: int = 3,
        min_alt_chars: int = 12
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        self.min_pixel_area = min_pixel_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_alt_words = min_alt_words
        self.min_alt_chars = min_alt_chars

        self._compiled_alt_regexes = [re.compile(p, re.IGNORECASE) for p in self.GENERIC_ALT_PATTERNS]
        self._compiled_url_regexes = [re.compile(p, re.IGNORECASE) for p in self.DISALLOWED_URL_PATTERNS]

    def validate_url(self, image_url: str) -> Tuple[bool, str]:
        """Checks protocol, extension, and tracking/ad indicators in image URL."""
        if not image_url or not isinstance(image_url, str):
            return False, "empty_url"

        parsed = urlparse(image_url)
        if parsed.scheme not in ("http", "https"):
            return False, f"invalid_scheme_{parsed.scheme}"

        lower_path = parsed.path.lower()
        # Check disallowed tracking patterns
        for reg in self._compiled_url_regexes:
            if reg.search(image_url):
                return False, "tracking_or_ad_pixel_url"

        # Check file extension if path has one
        if "." in lower_path:
            ext = "." + lower_path.rsplit(".", 1)[-1]
            if ext not in self.ALLOWED_EXTENSIONS:
                return False, f"unsupported_extension_{ext}"

        return True, "valid_url"

    def validate_dimensions(self, width: int, height: int) -> Tuple[bool, str]:
        """Validates spatial resolution bounds, aspect ratio, and total pixel area."""
        if width < self.min_width or height < self.min_height:
            return False, f"resolution_too_small_{width}x{height}"

        if width > self.max_width or height > self.max_height:
            return False, f"resolution_too_large_{width}x{height}"

        area = width * height
        if area < self.min_pixel_area:
            return False, f"insufficient_pixel_area_{area}"

        aspect_ratio = width / max(1, height)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, f"invalid_aspect_ratio_{aspect_ratio:.2f}"

        return True, "valid_dimensions"

    def validate_alt_text(self, alt_text: str) -> Tuple[bool, str]:
        """
        Filters out low-information, boilerplate, or empty alt-text descriptions.
        """
        if not alt_text:
            return False, "missing_alt_text"

        cleaned = alt_text.strip()
        if len(cleaned) < self.min_alt_chars:
            return False, f"alt_text_too_short_{len(cleaned)}_chars"

        words = cleaned.split()
        if len(words) < self.min_alt_words:
            return False, f"insufficient_words_{len(words)}"

        # Check generic boilerplate patterns
        for reg in self._compiled_alt_regexes:
            if reg.match(cleaned):
                return False, "generic_boilerplate_alt_text"

        return True, "valid_alt_text"

    def validate_candidate(
        self,
        image_url: str,
        alt_text: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation on a candidate image asset.
        """
        url_valid, url_reason = self.validate_url(image_url)
        alt_valid, alt_reason = self.validate_alt_text(alt_text)

        dim_valid = True
        dim_reason = "dimensions_omitted"
        if width is not None and height is not None:
            dim_valid, dim_reason = self.validate_dimensions(width, height)

        passed = url_valid and alt_valid and dim_valid

        return {
            "passed": passed,
            "url_valid": url_valid,
            "url_reason": url_reason,
            "alt_valid": alt_valid,
            "alt_reason": alt_reason,
            "dimensions_valid": dim_valid,
            "dimension_reason": dim_reason
        }
