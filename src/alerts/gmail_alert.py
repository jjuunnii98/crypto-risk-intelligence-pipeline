from __future__ import annotations

from typing import Iterable


def send_gmail_alert(subject: str, body: str, recipients: Iterable[str]) -> None:
    """
    Placeholder for Gmail alert sender.
    Later this can be integrated with SMTP or Gmail API.
    """
    recipient_str = ", ".join(recipients)
    print(f"[GMAIL ALERT] To={recipient_str} | Subject={subject}")
    print(body)