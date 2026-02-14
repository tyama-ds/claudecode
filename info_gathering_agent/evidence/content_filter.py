"""
Content Filter - Filter out ads, spam, and irrelevant content from search results.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
from urllib.parse import urlparse


# Known ad/tracking domains to block
AD_DOMAINS: Set[str] = {
    # Ad networks
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "googleads.g.doubleclick.net", "adservice.google.com",
    "ads.google.com", "pagead2.googlesyndication.com",
    "adsense.google.com", "adwords.google.com",
    "facebook.com/ads", "ads.facebook.com",
    "amazon-adsystem.com", "advertising.amazon.com",
    "ads.yahoo.com", "adtech.yahooinc.com",
    "adsserver.bing.com", "ads.microsoft.com",
    # Tracking/analytics as primary content
    "tracking.com", "click.email", "click.mail",
    "redirect.viglink.com", "go.skimresources.com",
    # Affiliate/comparison sites that are often ad-heavy
    "offers.comparegroups.com", "clickserve.dartsearch.net",
    "adfarm.mediaplex.com", "serving-sys.com",
    # Japanese ad networks
    "ad.shinobi.jp", "ads.microad.jp", "i-mobile.co.jp",
    "admatrix.jp", "ad-stir.com", "geniee.co.jp",
}

# URL patterns that indicate ad/tracking content
AD_URL_PATTERNS: List[str] = [
    r"/ads?/", r"/advert", r"/banner", r"/sponsor",
    r"/affiliate", r"/track(ing)?/", r"/click\?",
    r"/redir(ect)?", r"doubleclick", r"adsense",
    r"/promo(tion)?/", r"utm_source=", r"utm_campaign=",
    r"/partner/", r"/aff/", r"gclid=", r"fbclid=",
    r"/pixel", r"/beacon", r"googleads",
]

# Content patterns indicating ad/spam content
AD_CONTENT_PATTERNS: List[str] = [
    # Aggressive calls to action
    r"今すぐ(購入|申し込|クリック|登録|無料)",
    r"期間限定.{0,10}(セール|割引|キャンペーン)",
    r"(?:buy|order|subscribe|sign ?up) now",
    r"limited time offer",
    r"click here (to|for)",
    r"(無料|free).{0,10}(お試し|trial|download)",
    # Price/discount spam
    r"(?:最大|up to)?\s*\d{1,3}%\s*(?:off|オフ|割引)",
    r"(?:only|just|たったの)\s*[¥$€£]\s*\d+",
    r"(?:驚き|amazing|incredible)の(?:価格|price)",
    # Clickbait patterns
    r"(?:あなた|you)(?:だけ|も|は).{0,20}(?:秘密|secret|trick)",
    r"(?:医者|doctor|専門家)が.{0,20}(?:教えない|won't tell)",
    r"(?:衝撃|shocking|amazing).{0,20}(?:事実|truth|fact)",
    r"(?:一つ|one|this).{0,15}(?:トリック|trick|method)",
    # Spam email patterns
    r"(?:unsubscribe|配信停止|購読解除)",
    r"this (?:email|message) was sent",
    r"you(?:'re| are) receiving this",
    # Cookie/privacy banners
    r"(?:accept|同意).{0,30}(?:cookies?|クッキー)",
    r"privacy (?:policy|notice)",
    r"(?:we use|using) cookies",
    # Social sharing noise
    r"(?:share|シェア).{0,10}(?:on|via|で).{0,10}(?:twitter|facebook|line)",
    r"(?:follow|フォロー).{0,10}(?:us|me|私)",
    r"(?:like|いいね).{0,10}(?:on|で).{0,10}(?:facebook|instagram)",
    # Navigation/boilerplate
    r"(?:related|関連).{0,10}(?:articles?|記事|posts?)",
    r"(?:you may also|こちらも).{0,10}(?:like|おすすめ)",
    r"(?:popular|人気).{0,10}(?:posts?|記事)",
    r"(?:prev|next|前|次)(?:ious)?.{0,5}(?:article|記事|post)",
]

# Patterns indicating affiliate/commercial bias
COMMERCIAL_BIAS_PATTERNS: List[str] = [
    r"(?:affiliate|アフィリエイト).{0,30}(?:link|リンク|disclaimer)",
    r"(?:commission|手数料).{0,30}(?:earn|得)",
    r"(?:sponsored|スポンサー).{0,20}(?:content|post|記事)",
    r"(?:広告|advertisement|PR|ad)(?:\s*[:：]|\s*\|)",
    r"(?:amazon|楽天|affiliate).{0,20}(?:アソシエイト|associate)",
    r"this (?:article|post) (?:contains|includes) affiliate",
    r"(?:本記事|この記事).{0,20}(?:広告|PR|プロモーション)",
]

# Low-quality domain patterns
LOW_QUALITY_DOMAIN_PATTERNS: List[str] = [
    r"free.{0,10}(?:download|ダウンロード)",
    r"(?:coupon|クーポン|deal|セール)s?\.(?:com|jp|net)",
    r"(?:compare|比較|review|レビュー)s?\.(?:com|jp|net)",
    r"(?:cheap|安い|格安)\.(?:com|jp|net)",
    r"(?:best|top|ranking|ランキング)\d*\.(?:com|jp|net)",
]


@dataclass
class FilterResult:
    """Result of content filtering."""
    should_include: bool
    reason: str = ""
    quality_score: float = 1.0  # 0-1, lower means more likely to be ads/spam
    flags: List[str] = field(default_factory=list)


@dataclass
class ContentFilterConfig:
    """Configuration for content filtering."""
    # Enable/disable filtering
    enable_ad_domain_filter: bool = True
    enable_ad_url_filter: bool = True
    enable_ad_content_filter: bool = True
    enable_commercial_bias_filter: bool = True
    enable_low_quality_domain_filter: bool = True

    # Thresholds
    min_quality_score: float = 0.3  # Content below this score is filtered
    max_ad_pattern_matches: int = 3  # More than this many ad patterns = filter
    min_content_length: int = 100  # Minimum content length to consider

    # Custom blocklist (user-defined)
    custom_blocked_domains: List[str] = field(default_factory=list)
    custom_blocked_patterns: List[str] = field(default_factory=list)

    # Whitelist (always allow)
    whitelisted_domains: List[str] = field(default_factory=list)


class ContentFilter:
    """
    Filter out ads, spam, and low-quality content from search results.

    This filter is applied during content extraction to prevent
    irrelevant or promotional content from entering the research results.
    """

    def __init__(self, config: ContentFilterConfig = None):
        """
        Initialize ContentFilter.

        Args:
            config: Filter configuration (uses defaults if not provided)
        """
        self.config = config or ContentFilterConfig()

        # Compile patterns for efficiency
        self._ad_url_patterns = [re.compile(p, re.IGNORECASE) for p in AD_URL_PATTERNS]
        self._ad_content_patterns = [re.compile(p, re.IGNORECASE) for p in AD_CONTENT_PATTERNS]
        self._commercial_patterns = [re.compile(p, re.IGNORECASE) for p in COMMERCIAL_BIAS_PATTERNS]
        self._low_quality_domain_patterns = [
            re.compile(p, re.IGNORECASE) for p in LOW_QUALITY_DOMAIN_PATTERNS
        ]

        # Add custom patterns
        if self.config.custom_blocked_patterns:
            self._custom_patterns = [
                re.compile(p, re.IGNORECASE) for p in self.config.custom_blocked_patterns
            ]
        else:
            self._custom_patterns = []

    def filter_url(self, url: str) -> FilterResult:
        """
        Filter based on URL only.

        Args:
            url: URL to check

        Returns:
            FilterResult indicating whether to include this URL
        """
        flags = []
        quality_score = 1.0

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            full_url = url.lower()
        except Exception:
            return FilterResult(
                should_include=False,
                reason="Invalid URL format",
                quality_score=0.0,
                flags=["invalid_url"]
            )

        # Check whitelist first
        if self._is_whitelisted(domain):
            return FilterResult(
                should_include=True,
                reason="Whitelisted domain",
                quality_score=1.0
            )

        # Check ad domains
        if self.config.enable_ad_domain_filter:
            for ad_domain in AD_DOMAINS:
                if ad_domain in domain:
                    return FilterResult(
                        should_include=False,
                        reason=f"Blocked ad domain: {ad_domain}",
                        quality_score=0.0,
                        flags=["ad_domain"]
                    )

            # Check custom blocked domains
            for blocked in self.config.custom_blocked_domains:
                if blocked.lower() in domain:
                    return FilterResult(
                        should_include=False,
                        reason=f"Custom blocked domain: {blocked}",
                        quality_score=0.0,
                        flags=["custom_blocked_domain"]
                    )

        # Check ad URL patterns
        if self.config.enable_ad_url_filter:
            for pattern in self._ad_url_patterns:
                if pattern.search(full_url):
                    flags.append("ad_url_pattern")
                    quality_score -= 0.3
                    break

        # Check low-quality domain patterns
        if self.config.enable_low_quality_domain_filter:
            for pattern in self._low_quality_domain_patterns:
                if pattern.search(domain):
                    flags.append("low_quality_domain")
                    quality_score -= 0.2
                    break

        should_include = quality_score >= self.config.min_quality_score
        reason = "" if should_include else f"Low URL quality score: {quality_score:.2f}"

        return FilterResult(
            should_include=should_include,
            reason=reason,
            quality_score=max(0.0, quality_score),
            flags=flags
        )

    def filter_content(
        self,
        url: str,
        title: str,
        content: str,
    ) -> FilterResult:
        """
        Filter based on URL and content.

        Args:
            url: Source URL
            title: Page title
            content: Page content

        Returns:
            FilterResult indicating whether to include this content
        """
        flags = []
        quality_score = 1.0

        # First check URL
        url_result = self.filter_url(url)
        if not url_result.should_include:
            return url_result

        flags.extend(url_result.flags)
        quality_score = url_result.quality_score

        # Check content length
        if len(content) < self.config.min_content_length:
            return FilterResult(
                should_include=False,
                reason=f"Content too short: {len(content)} chars",
                quality_score=0.1,
                flags=["short_content"]
            )

        # Check ad content patterns
        if self.config.enable_ad_content_filter:
            ad_matches = 0
            for pattern in self._ad_content_patterns:
                if pattern.search(content) or pattern.search(title):
                    ad_matches += 1
                    if ad_matches == 1:
                        flags.append("ad_content")

            if ad_matches > self.config.max_ad_pattern_matches:
                return FilterResult(
                    should_include=False,
                    reason=f"Too many ad patterns matched: {ad_matches}",
                    quality_score=0.1,
                    flags=flags + [f"ad_patterns_{ad_matches}"]
                )

            # Deduct quality score based on ad matches
            quality_score -= ad_matches * 0.1

        # Check commercial bias patterns
        if self.config.enable_commercial_bias_filter:
            commercial_matches = 0
            for pattern in self._commercial_patterns:
                if pattern.search(content) or pattern.search(title):
                    commercial_matches += 1
                    if commercial_matches == 1:
                        flags.append("commercial_bias")

            if commercial_matches > 2:
                quality_score -= 0.3
                flags.append("high_commercial_bias")
            elif commercial_matches > 0:
                quality_score -= commercial_matches * 0.1

        # Check custom patterns
        for pattern in self._custom_patterns:
            if pattern.search(content) or pattern.search(title):
                flags.append("custom_pattern")
                quality_score -= 0.2
                break

        # Calculate final decision
        quality_score = max(0.0, min(1.0, quality_score))
        should_include = quality_score >= self.config.min_quality_score

        reason = ""
        if not should_include:
            reason = f"Low quality score ({quality_score:.2f}): {', '.join(flags)}"

        return FilterResult(
            should_include=should_include,
            reason=reason,
            quality_score=quality_score,
            flags=flags
        )

    def _is_whitelisted(self, domain: str) -> bool:
        """Check if domain is in whitelist."""
        for whitelisted in self.config.whitelisted_domains:
            if whitelisted.lower() in domain:
                return True
        return False

    def analyze_content_quality(
        self,
        content: str,
        title: str = "",
    ) -> Tuple[float, List[str]]:
        """
        Analyze content quality without making a filter decision.

        Useful for getting quality metrics without filtering.

        Args:
            content: Content to analyze
            title: Optional title

        Returns:
            Tuple of (quality_score, list_of_issues)
        """
        issues = []
        quality_score = 1.0

        # Check content length
        word_count = len(content.split())
        if word_count < 50:
            issues.append("very_short_content")
            quality_score -= 0.3
        elif word_count < 100:
            issues.append("short_content")
            quality_score -= 0.1

        # Check for excessive punctuation (clickbait indicator)
        exclaim_count = content.count("!") + content.count("！")
        if exclaim_count > 10:
            issues.append("excessive_exclamations")
            quality_score -= 0.2

        # Check for ad patterns
        ad_count = sum(
            1 for p in self._ad_content_patterns
            if p.search(content) or p.search(title)
        )
        if ad_count > 0:
            issues.append(f"ad_patterns_{ad_count}")
            quality_score -= ad_count * 0.08

        # Check for commercial bias
        commercial_count = sum(
            1 for p in self._commercial_patterns
            if p.search(content) or p.search(title)
        )
        if commercial_count > 0:
            issues.append(f"commercial_bias_{commercial_count}")
            quality_score -= commercial_count * 0.1

        # Check for excessive links/repetition (often spam indicator)
        link_pattern = re.compile(r'https?://', re.IGNORECASE)
        link_count = len(link_pattern.findall(content))
        if link_count > 20:
            issues.append("excessive_links")
            quality_score -= 0.2

        return max(0.0, min(1.0, quality_score)), issues

    def get_blocked_domains(self) -> Set[str]:
        """Get all blocked domains including custom ones."""
        blocked = AD_DOMAINS.copy()
        blocked.update(self.config.custom_blocked_domains)
        return blocked

    def add_blocked_domain(self, domain: str) -> None:
        """Add a domain to the blocklist."""
        self.config.custom_blocked_domains.append(domain.lower())

    def add_whitelisted_domain(self, domain: str) -> None:
        """Add a domain to the whitelist."""
        self.config.whitelisted_domains.append(domain.lower())

    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a custom pattern to block."""
        self.config.custom_blocked_patterns.append(pattern)
        self._custom_patterns.append(re.compile(pattern, re.IGNORECASE))


def create_strict_filter() -> ContentFilter:
    """Create a filter with strict settings for high-quality research."""
    config = ContentFilterConfig(
        enable_ad_domain_filter=True,
        enable_ad_url_filter=True,
        enable_ad_content_filter=True,
        enable_commercial_bias_filter=True,
        enable_low_quality_domain_filter=True,
        min_quality_score=0.5,
        max_ad_pattern_matches=2,
        min_content_length=200,
    )
    return ContentFilter(config)


def create_moderate_filter() -> ContentFilter:
    """Create a filter with moderate settings."""
    config = ContentFilterConfig(
        enable_ad_domain_filter=True,
        enable_ad_url_filter=True,
        enable_ad_content_filter=True,
        enable_commercial_bias_filter=False,
        enable_low_quality_domain_filter=True,
        min_quality_score=0.3,
        max_ad_pattern_matches=3,
        min_content_length=100,
    )
    return ContentFilter(config)


def create_minimal_filter() -> ContentFilter:
    """Create a filter with minimal settings (only blocks obvious ads)."""
    config = ContentFilterConfig(
        enable_ad_domain_filter=True,
        enable_ad_url_filter=False,
        enable_ad_content_filter=False,
        enable_commercial_bias_filter=False,
        enable_low_quality_domain_filter=False,
        min_quality_score=0.1,
        max_ad_pattern_matches=10,
        min_content_length=50,
    )
    return ContentFilter(config)
