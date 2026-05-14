"""External data source connectors for MemoryMesh.

Each connector fetches data from an external protocol or local file and
converts it into :class:`~memorymesh.core.models.ParsedDocument` objects
that the normal chunking + embedding pipeline can consume.

Available connectors
--------------------
* :mod:`memorymesh.connectors.imap_connector` — IMAP email source (stdlib only).
* :mod:`memorymesh.connectors.caldav_connector` — CalDAV calendar source
  (requires ``caldav`` package).
* :mod:`memorymesh.connectors.browser_history_connector` — Chrome / Firefox /
  Edge local SQLite history (stdlib only).
* :mod:`memorymesh.connectors.kindle_connector` — Kindle ``My Clippings.txt``
  highlights (stdlib only).
* :mod:`memorymesh.connectors.rss_connector` — RSS / Atom feeds
  (requires ``feedparser`` package).
* :mod:`memorymesh.connectors.zotero_connector` — Zotero local SQLite database
  (stdlib only).
* :mod:`memorymesh.connectors.whatsapp_connector` — WhatsApp chat export
  ``.txt`` files, Android and iOS formats (stdlib only).
* :mod:`memorymesh.connectors.telegram_connector` — Telegram Desktop
  ``result.json`` export (stdlib only).
* :mod:`memorymesh.connectors.twitter_connector` — Twitter/X data archive
  ``tweets.js`` / ``likes.js``, supports ``.zip`` input (stdlib only).
* :mod:`memorymesh.connectors.spotify_connector` — Spotify extended streaming
  history ``Streaming_History_Audio_*.json`` files (stdlib only).
* :mod:`memorymesh.connectors.notion_connector` — Notion workspace pages and
  database entries via the Notion API (stdlib urllib only).
* :mod:`memorymesh.connectors.github_connector` — GitHub repository issues and
  pull requests via the GitHub REST API (stdlib urllib only).
* :mod:`memorymesh.connectors.readwise_connector` — Readwise book highlights
  via the Readwise API v2 (stdlib urllib only).
* :mod:`memorymesh.connectors.slack_connector` — Slack channel messages via
  the Slack Web API (stdlib urllib only).
* :mod:`memorymesh.connectors.apple_health_connector` — Apple Health
  ``export.xml`` / ``export.zip`` records and workouts (stdlib only).
* :mod:`memorymesh.connectors.google_location_connector` — Google Maps
  location history from Google Takeout ``Records.json`` or Semantic
  Location History (stdlib only).
* :mod:`memorymesh.connectors.strava_connector` — Strava activities via
  the Strava API v3 with OAuth token auto-refresh (stdlib urllib only).
* :mod:`memorymesh.connectors.bank_csv_connector` — Bank/credit card
  transaction CSV exports with auto-detected column names (stdlib only).
* :mod:`memorymesh.connectors.discord_connector` — DiscordChatExporter
  JSON exports, grouped by channel-day (stdlib only).
* :mod:`memorymesh.connectors.linkedin_connector` — LinkedIn data export
  CSVs: connections, messages, posts (stdlib only).
* :mod:`memorymesh.connectors.youtube_connector` — YouTube watch history
  from Google Takeout ``watch-history.json`` (stdlib only).
* :mod:`memorymesh.connectors.facebook_connector` — Facebook/Meta data
  archive JSON, including posts and message threads (stdlib only).
* :mod:`memorymesh.connectors.instagram_connector` — Instagram data
  archive JSON, including posts and message threads (stdlib only).
* :mod:`memorymesh.connectors.pinterest_connector` — Pinterest data
  export CSV, grouped by board (stdlib only).
* :mod:`memorymesh.connectors.goodreads_connector` — Goodreads library
  export CSV with optional shelf filtering (stdlib only).
* :mod:`memorymesh.connectors.anki_connector` — Anki flashcard
  ``collection.anki2`` SQLite database (stdlib only).
* :mod:`memorymesh.connectors.logseq_connector` — Logseq graph Markdown
  files with block-syntax stripping (stdlib only).
* :mod:`memorymesh.connectors.letterboxd_connector` — Letterboxd data
  export CSVs: diary, reviews, and watchlist (stdlib only).
* :mod:`memorymesh.connectors.reddit_connector` — Reddit public JSON API:
  submitted posts and comments for a username (no OAuth, stdlib urllib
  only).
* :mod:`memorymesh.connectors.lastfm_connector` — Last.fm scrobble
  history via the Last.fm API, grouped by calendar month (stdlib urllib
  only).
* :mod:`memorymesh.connectors.steam_connector` — Steam game library and
  playtime via the Steam Web API; optional achievement counts and
  recently-played summary (stdlib urllib only).
* :mod:`memorymesh.connectors.hackernews_connector` — HackerNews
  stories and comments via the Algolia HN API (no key required, stdlib
  urllib only).
* :mod:`memorymesh.connectors.pocket_connector` — Pocket saved articles
  via the Pocket v3 API with consumer-key + access-token auth (stdlib
  urllib only).
* :mod:`memorymesh.connectors.linear_connector` — Linear issues assigned
  to the authenticated user via the Linear GraphQL API (stdlib urllib
  only).
* :mod:`memorymesh.connectors.jira_connector` — Jira Cloud issues via
  the Jira REST API v3 with Basic auth (stdlib urllib only).
* :mod:`memorymesh.connectors.confluence_connector` — Confluence Cloud
  pages via the Confluence REST API with Basic auth (stdlib urllib only).
* :mod:`memorymesh.connectors.gitlab_connector` — GitLab issues and merge
  requests via the GitLab REST API with private-token auth (stdlib urllib
  only).
* :mod:`memorymesh.connectors.trello_connector` — Trello cards from all
  accessible boards via the Trello REST API (stdlib urllib only).
* :mod:`memorymesh.connectors.asana_connector` — Asana tasks from
  workspaces or projects via the Asana REST API (stdlib urllib only).
* :mod:`memorymesh.connectors.roam_connector` — Roam Research graph JSON
  export with recursive block flattening (stdlib only).
* :mod:`memorymesh.connectors.apple_notes_connector` — Apple Notes from
  the local ``NoteStore.sqlite`` database (macOS + stdlib only).
* :mod:`memorymesh.connectors.mastodon_connector` — Mastodon toots via
  the Mastodon API with Link-header pagination (stdlib urllib only).
* :mod:`memorymesh.connectors.bluesky_connector` — Bluesky posts via the
  AT Protocol API with session auth (stdlib urllib only).
* :mod:`memorymesh.connectors.chesscom_connector` — Chess.com monthly
  game archives via the public Chess.com API (no auth, stdlib urllib only).
* :mod:`memorymesh.connectors.duolingo_connector` — Duolingo language
  learning data from a JSON export (stdlib only).
* :mod:`memorymesh.connectors.feedly_connector` — Feedly RSS reader
  articles via the Feedly API with continuation pagination (stdlib urllib
  only).
* :mod:`memorymesh.connectors.hypothesis_connector` — Hypothes.is web
  annotations via the Hypothes.is API (stdlib urllib only).
* :mod:`memorymesh.connectors.garmin_connector` — Garmin Connect activity
  summaries from a data export ZIP or directory (stdlib only).
* :mod:`memorymesh.connectors.airtable_connector` — Airtable base records
  via the Airtable REST API with personal access token (stdlib urllib only).
"""
