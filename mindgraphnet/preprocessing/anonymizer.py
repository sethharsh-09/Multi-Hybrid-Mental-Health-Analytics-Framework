"""
MindGraphNet - Privacy & Data Anonymization Engine
Implements irreversible salted SHA-256 pseudonymization and multi-pattern PII scrubbing.
"""
import re
import hashlib
from config.settings import SALT

class DataAnonymizer:
    def __init__(self, salt: str = SALT):
        self.salt = salt

    def pseudonymize_identifier(self, identifier: str) -> str:
        """
        Generates a 16-character salted SHA-256 pseudonym for author IDs.
        """
        if not identifier:
            return "anon_unknown"
        hasher = hashlib.sha256()
        hasher.update((str(identifier) + self.salt).encode("utf-8"))
        return f"usr_{hasher.hexdigest()[:12]}"

    def scrub_pii(self, text: str) -> str:
        """
        Redacts personally identifiable information (PII) including emails,
        phone numbers, URLs, user handles, and IP addresses.
        """
        if not isinstance(text, str):
            return ""
        # 1. Emails
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "<REDACTED_EMAIL>", text)
        # 2. URLs
        text = re.sub(r"https?://\S+|www\.\S+", "<REDACTED_URL>", text)
        # 3. Phone Numbers (various international / standard formats)
        text = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "<REDACTED_PHONE>", text)
        # 4. User mentions (u/username or @username)
        text = re.sub(r"(?:u/|@)[\w\-_]+", "<REDACTED_USER>", text)
        # 5. IPv4 addresses
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<REDACTED_IP>", text)
        return text
