"""Récupération des transcriptions (scripts) de vidéos YouTube.

Première brique de l'ingestion : à partir d'une URL ou d'un identifiant de vidéo,
on récupère le texte parlé de la vidéo. Ce texte servira plus tard à être résumé
par le LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


# Langues recherchées par ordre de préférence. On essaie le français d'abord,
# puis l'anglais (la plupart des chaînes IA publient en anglais).
LANGUES_PREFEREES = ("fr", "en")


@dataclass
class Transcription:
    """Le résultat d'une récupération réussie."""

    video_id: str
    texte: str
    nb_segments: int


def extraire_video_id(url_ou_id: str) -> str:
    """Extrait l'identifiant d'une vidéo à partir d'une URL YouTube.

    Accepte aussi bien une URL complète qu'un identifiant déjà nu.
    Exemples gérés :
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://youtu.be/dQw4w9WgXcQ
        dQw4w9WgXcQ
    """
    # Un identifiant YouTube fait 11 caractères (lettres, chiffres, - et _).
    motif_id = r"[0-9A-Za-z_-]{11}"

    # Si on nous a déjà passé un identifiant nu, on le renvoie tel quel.
    if re.fullmatch(motif_id, url_ou_id):
        return url_ou_id

    # Sinon on cherche le paramètre v=... ou le format court youtu.be/...
    motifs = [
        rf"v=({motif_id})",
        rf"youtu\.be/({motif_id})",
        rf"/embed/({motif_id})",
    ]
    for motif in motifs:
        resultat = re.search(motif, url_ou_id)
        if resultat:
            return resultat.group(1)

    raise ValueError(f"Impossible d'extraire l'identifiant de la vidéo : {url_ou_id!r}")


def recuperer_transcription(
    url_ou_id: str,
    langues: tuple[str, ...] = LANGUES_PREFEREES,
) -> Transcription:
    """Récupère la transcription complète d'une vidéo sous forme de texte.

    Lève une ValueError avec un message clair si la transcription est
    indisponible (sous-titres désactivés, vidéo privée, etc.).
    """
    video_id = extraire_video_id(url_ou_id)

    api = YouTubeTranscriptApi()
    try:
        resultat = api.fetch(video_id, languages=langues)
    except TranscriptsDisabled:
        raise ValueError(f"Les sous-titres sont désactivés pour la vidéo {video_id}.")
    except NoTranscriptFound:
        raise ValueError(
            f"Aucune transcription trouvée pour {video_id} dans les langues {langues}."
        )
    except VideoUnavailable:
        raise ValueError(f"La vidéo {video_id} est indisponible (privée ou supprimée).")

    # `resultat` est itérable : chaque segment a un attribut .text.
    # On assemble tous les segments en un seul texte continu.
    texte = " ".join(segment.text for segment in resultat)

    return Transcription(
        video_id=video_id, texte=texte, nb_segments=len(resultat.to_raw_data())
    )


if __name__ == "__main__":
    # Petit test manuel : remplace l'URL par une vraie vidéo pour essayer.
    #   uv run python -m app.ingestion.youtube
    url_test = "https://www.youtube.com/watch?v=R7O2TaM709Y"
    transcription = recuperer_transcription(url_test)
    print(f"Vidéo      : {transcription.video_id}")
    print(f"Segments   : {transcription.nb_segments}")
    print(f"Longueur   : {len(transcription.texte)} caractères")
    print("---")
    print(transcription.texte[:5000], "...")
