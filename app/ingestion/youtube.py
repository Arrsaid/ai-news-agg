"""Ingestion YouTube : suivi de chaînes et récupération des transcriptions.

À partir d'une chaîne (identifiant, URL ou handle "@nom"), on liste les vidéos
publiées récemment, puis on récupère leur transcription (script). Ce texte
servira plus tard à être résumé par le LLM.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import feedparser
from pydantic import BaseModel
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)


# Langues recherchées par ordre de préférence : le français d'abord, puis
# l'anglais (la plupart des chaînes IA publient en anglais).
LANGUES_PREFEREES = ("fr", "en")

# Motifs d'identifiants : une vidéo fait 11 caractères, une chaîne "UC" + 22.
_MOTIF_VIDEO_ID = r"[0-9A-Za-z_-]{11}"
_MOTIF_CHANNEL_ID = r"UC[0-9A-Za-z_-]{22}"


class Transcript(BaseModel):
    """La transcription (texte parlé) d'une vidéo."""

    text: str


class ChannelVideo(BaseModel):
    """Une vidéo publiée par une chaîne suivie."""

    title: str
    url: str
    video_id: str
    published_at: datetime  # en UTC
    description: str
    transcript: Optional[str] = None


class YouTubeScraper:
    """Récupère les vidéos récentes d'une chaîne et leurs transcriptions."""

    def __init__(self, langues: tuple[str, ...] = LANGUES_PREFEREES):
        self.langues = langues
        self.transcript_api = YouTubeTranscriptApi()

    # --- Helpers internes ---------------------------------------------------

    def _get_rss_url(self, channel_id: str) -> str:
        """URL du flux RSS public d'une chaîne (les ~15 vidéos les plus récentes)."""
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    def _extract_video_id(self, url_or_id: str) -> str:
        """Extrait l'identifiant d'une vidéo depuis une URL (ou le renvoie tel quel)."""
        # Identifiant déjà nu.
        if re.fullmatch(_MOTIF_VIDEO_ID, url_or_id):
            return url_or_id
        # Formats : watch?v=..., youtu.be/..., /embed/..., /shorts/...
        for motif in (
            rf"v=({_MOTIF_VIDEO_ID})",
            rf"youtu\.be/({_MOTIF_VIDEO_ID})",
            rf"/embed/({_MOTIF_VIDEO_ID})",
            rf"/shorts/({_MOTIF_VIDEO_ID})",
        ):
            trouve = re.search(motif, url_or_id)
            if trouve:
                return trouve.group(1)
        raise ValueError(f"Impossible d'extraire l'identifiant de la vidéo : {url_or_id!r}")

    def _resolve_channel_id(self, channel: str) -> str:
        """Renvoie l'identifiant UC... d'une chaîne (id nu, URL, ou handle "@nom")."""
        channel = channel.strip()

        # Identifiant nu, ou déjà présent dans une URL ".../channel/UC...".
        if re.fullmatch(_MOTIF_CHANNEL_ID, channel):
            return channel
        trouve = re.search(rf"channel/({_MOTIF_CHANNEL_ID})", channel)
        if trouve:
            return trouve.group(1)

        # Sinon c'est un handle/nom : on lit la page et on récupère son lien
        # canonique (".../channel/UC..."). Le handle est encodé car il peut
        # contenir des accents.
        if not channel.startswith("http"):
            handle = channel if channel.startswith("@") else f"@{channel}"
            channel = "https://www.youtube.com/" + urllib.parse.quote(handle)
        requete = urllib.request.Request(channel, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(requete, timeout=15) as reponse:
            html = reponse.read().decode("utf-8", errors="ignore")

        trouve = re.search(rf"channel/({_MOTIF_CHANNEL_ID})", html)
        if not trouve:
            raise ValueError(f"Identifiant de chaîne introuvable pour {channel!r}.")
        return trouve.group(1)

    # --- API publique -------------------------------------------------------

    def get_transcript(self, video_id: str) -> Optional[Transcript]:
        """Récupère la transcription d'une vidéo, ou None si indisponible."""
        try:
            fetched = self.transcript_api.fetch(video_id, languages=self.langues)
            text = " ".join(snippet.text for snippet in fetched.snippets)
            return Transcript(text=text)
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception:
            # Erreur réseau ou autre : on n'interrompt pas le traitement du lot.
            return None

    def get_latest_videos(
        self,
        channel: str,
        depuis: datetime,
        jusqu_a: Optional[datetime] = None,
    ) -> list[ChannelVideo]:
        """Vidéos d'une chaîne publiées entre `depuis` et `jusqu_a`.

        `channel` peut être un identifiant, une URL ou un handle "@nom".
        `depuis` et `jusqu_a` doivent être timezone-aware (`jusqu_a` vaut
        « maintenant » par défaut). Les Shorts sont ignorés.
        """
        jusqu_a = jusqu_a or datetime.now(timezone.utc)
        channel_id = self._resolve_channel_id(channel)
        feed = feedparser.parse(self._get_rss_url(channel_id))

        videos: list[ChannelVideo] = []
        for entry in feed.entries:
            if "/shorts/" in entry.link:
                continue
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if depuis <= published_at <= jusqu_a:
                videos.append(
                    ChannelVideo(
                        title=entry.title,
                        url=entry.link,
                        video_id=self._extract_video_id(entry.link),
                        published_at=published_at,
                        description=entry.get("summary", ""),
                    )
                )

        videos.sort(key=lambda v: v.published_at, reverse=True)
        return videos

    def scrape_channel(
        self,
        channel: str,
        depuis: datetime,
        jusqu_a: Optional[datetime] = None,
    ) -> list[ChannelVideo]:
        """Liste les vidéos d'une chaîne dans un intervalle, avec transcription."""
        videos = self.get_latest_videos(channel, depuis, jusqu_a)
        result: list[ChannelVideo] = []
        for video in videos:
            transcript = self.get_transcript(video.video_id)
            result.append(
                video.model_copy(
                    update={"transcript": transcript.text if transcript else None}
                )
            )
        return result


if __name__ == "__main__":
    # Démo : vidéos publiées au cours des 30 derniers jours.
    #   uv run python -m app.ingestion.youtube
    from datetime import timedelta

    scraper = YouTubeScraper()
    depuis = datetime.now(timezone.utc) - timedelta(days=30)

    for video in scraper.scrape_channel("@QuantaScienceChannel", depuis=depuis):
        print(f"{video.published_at:%Y-%m-%d}  {video.title}")
        print(f"            {video.url}")
        if video.transcript:
            print(f"            {video.transcript[:200]} ...")
