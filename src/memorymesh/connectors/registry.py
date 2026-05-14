"""Connector registry for MemoryMesh.

Maps connector type strings (as used in ``config.yaml``) to their
``(ConfigClass, ConnectorClass)`` pairs so the ``sync`` CLI command can
instantiate any connector from a plain dict of settings.

Usage
-----
::

    from memorymesh.connectors.registry import CONNECTOR_REGISTRY

    cfg_cls, conn_cls = CONNECTOR_REGISTRY["jira"]
    config_obj = cfg_cls(**raw_config_dict)
    connector = conn_cls(config_obj)
    for doc in connector.fetch_documents():
        ...
"""

from __future__ import annotations

from typing import Any

# Maps type string -> (ConfigClass, ConnectorClass).
# Populated lazily via _register() to keep import cost near zero until needed.
CONNECTOR_REGISTRY: dict[str, tuple[Any, Any]] = {}


def _register() -> None:
    """Populate CONNECTOR_REGISTRY with all known connectors."""
    from memorymesh.connectors.airtable_connector import (
        AirtableConfig,
        AirtableConnector,
    )
    from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector
    from memorymesh.connectors.apple_health_connector import (
        AppleHealthConfig,
        AppleHealthConnector,
    )
    from memorymesh.connectors.apple_notes_connector import (
        AppleNotesConfig,
        AppleNotesConnector,
    )
    from memorymesh.connectors.asana_connector import AsanaConfig, AsanaConnector
    from memorymesh.connectors.bank_csv_connector import BankCSVConfig, BankCSVConnector
    from memorymesh.connectors.bluesky_connector import BlueSkyConfig, BlueSkyConnector
    from memorymesh.connectors.browser_history_connector import (
        BrowserHistoryConfig,
        BrowserHistoryConnector,
    )
    from memorymesh.connectors.caldav_connector import CalDAVConfig, CalDAVConnector
    from memorymesh.connectors.chesscom_connector import (
        ChessComConfig,
        ChessComConnector,
    )
    from memorymesh.connectors.confluence_connector import (
        ConfluenceConfig,
        ConfluenceConnector,
    )
    from memorymesh.connectors.discord_connector import (
        DiscordConfig,
        DiscordConnector,
    )
    from memorymesh.connectors.duolingo_connector import (
        DuolingoConfig,
        DuolingoConnector,
    )
    from memorymesh.connectors.facebook_connector import (
        FacebookConfig,
        FacebookConnector,
    )
    from memorymesh.connectors.feedly_connector import FeedlyConfig, FeedlyConnector
    from memorymesh.connectors.garmin_connector import GarminConfig, GarminConnector
    from memorymesh.connectors.github_connector import GitHubConfig, GitHubConnector
    from memorymesh.connectors.gitlab_connector import GitLabConfig, GitLabConnector
    from memorymesh.connectors.goodreads_connector import (
        GoodreadsConfig,
        GoodreadsConnector,
    )
    from memorymesh.connectors.google_location_connector import (
        GoogleLocationConfig,
        GoogleLocationConnector,
    )
    from memorymesh.connectors.hackernews_connector import (
        HackerNewsConfig,
        HackerNewsConnector,
    )
    from memorymesh.connectors.hypothesis_connector import (
        HypothesisConfig,
        HypothesisConnector,
    )
    from memorymesh.connectors.imap_connector import IMAPConfig, IMAPConnector
    from memorymesh.connectors.instagram_connector import (
        InstagramConfig,
        InstagramConnector,
    )
    from memorymesh.connectors.jira_connector import JiraConfig, JiraConnector
    from memorymesh.connectors.kindle_connector import (
        KindleConfig,
        KindleConnector,
    )
    from memorymesh.connectors.lastfm_connector import LastFmConfig, LastFmConnector
    from memorymesh.connectors.letterboxd_connector import (
        LetterboxdConfig,
        LetterboxdConnector,
    )
    from memorymesh.connectors.linear_connector import LinearConfig, LinearConnector
    from memorymesh.connectors.linkedin_connector import (
        LinkedInConfig,
        LinkedInConnector,
    )
    from memorymesh.connectors.logseq_connector import LogseqConfig, LogseqConnector
    from memorymesh.connectors.mastodon_connector import (
        MastodonConfig,
        MastodonConnector,
    )
    from memorymesh.connectors.notion_connector import NotionConfig, NotionConnector
    from memorymesh.connectors.pinterest_connector import (
        PinterestConfig,
        PinterestConnector,
    )
    from memorymesh.connectors.pocket_connector import PocketConfig, PocketConnector
    from memorymesh.connectors.readwise_connector import (
        ReadwiseConfig,
        ReadwiseConnector,
    )
    from memorymesh.connectors.reddit_connector import RedditConfig, RedditConnector
    from memorymesh.connectors.roam_connector import RoamConfig, RoamConnector
    from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector
    from memorymesh.connectors.slack_connector import SlackConfig, SlackConnector
    from memorymesh.connectors.spotify_connector import (
        SpotifyConfig,
        SpotifyConnector,
    )
    from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector
    from memorymesh.connectors.strava_connector import StravaConfig, StravaConnector
    from memorymesh.connectors.telegram_connector import (
        TelegramConfig,
        TelegramConnector,
    )
    from memorymesh.connectors.trello_connector import TrelloConfig, TrelloConnector
    from memorymesh.connectors.twitter_connector import (
        TwitterConfig,
        TwitterConnector,
    )
    from memorymesh.connectors.whatsapp_connector import (
        WhatsAppConfig,
        WhatsAppConnector,
    )
    from memorymesh.connectors.youtube_connector import (
        YouTubeConfig,
        YouTubeConnector,
    )
    from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

    CONNECTOR_REGISTRY.update(
        {
            "airtable": (AirtableConfig, AirtableConnector),
            "anki": (AnkiConfig, AnkiConnector),
            "apple_health": (AppleHealthConfig, AppleHealthConnector),
            "apple_notes": (AppleNotesConfig, AppleNotesConnector),
            "asana": (AsanaConfig, AsanaConnector),
            "bank_csv": (BankCSVConfig, BankCSVConnector),
            "bluesky": (BlueSkyConfig, BlueSkyConnector),
            "browser_history": (BrowserHistoryConfig, BrowserHistoryConnector),
            "caldav": (CalDAVConfig, CalDAVConnector),
            "chesscom": (ChessComConfig, ChessComConnector),
            "confluence": (ConfluenceConfig, ConfluenceConnector),
            "discord": (DiscordConfig, DiscordConnector),
            "duolingo": (DuolingoConfig, DuolingoConnector),
            "facebook": (FacebookConfig, FacebookConnector),
            "feedly": (FeedlyConfig, FeedlyConnector),
            "garmin": (GarminConfig, GarminConnector),
            "github": (GitHubConfig, GitHubConnector),
            "gitlab": (GitLabConfig, GitLabConnector),
            "goodreads": (GoodreadsConfig, GoodreadsConnector),
            "google_location": (GoogleLocationConfig, GoogleLocationConnector),
            "hackernews": (HackerNewsConfig, HackerNewsConnector),
            "hypothesis": (HypothesisConfig, HypothesisConnector),
            "imap": (IMAPConfig, IMAPConnector),
            "instagram": (InstagramConfig, InstagramConnector),
            "jira": (JiraConfig, JiraConnector),
            "kindle": (KindleConfig, KindleConnector),
            "lastfm": (LastFmConfig, LastFmConnector),
            "letterboxd": (LetterboxdConfig, LetterboxdConnector),
            "linear": (LinearConfig, LinearConnector),
            "linkedin": (LinkedInConfig, LinkedInConnector),
            "logseq": (LogseqConfig, LogseqConnector),
            "mastodon": (MastodonConfig, MastodonConnector),
            "notion": (NotionConfig, NotionConnector),
            "pinterest": (PinterestConfig, PinterestConnector),
            "pocket": (PocketConfig, PocketConnector),
            "readwise": (ReadwiseConfig, ReadwiseConnector),
            "reddit": (RedditConfig, RedditConnector),
            "roam": (RoamConfig, RoamConnector),
            "rss": (RSSConfig, RSSConnector),
            "slack": (SlackConfig, SlackConnector),
            "spotify": (SpotifyConfig, SpotifyConnector),
            "steam": (SteamConfig, SteamConnector),
            "strava": (StravaConfig, StravaConnector),
            "telegram": (TelegramConfig, TelegramConnector),
            "trello": (TrelloConfig, TrelloConnector),
            "twitter": (TwitterConfig, TwitterConnector),
            "whatsapp": (WhatsAppConfig, WhatsAppConnector),
            "youtube": (YouTubeConfig, YouTubeConnector),
            "zotero": (ZoteroConfig, ZoteroConnector),
        }
    )


def get_connector_classes(connector_type: str) -> tuple[Any, Any]:
    """Return ``(ConfigClass, ConnectorClass)`` for *connector_type*.

    Populates the registry on first call.

    Args:
        connector_type: Connector type identifier string.

    Returns:
        ``(ConfigClass, ConnectorClass)`` tuple.

    Raises:
        KeyError: If *connector_type* is not registered.
    """
    if not CONNECTOR_REGISTRY:
        _register()
    return CONNECTOR_REGISTRY[connector_type]
