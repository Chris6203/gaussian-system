"""Simple file-based user authentication."""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Optional

import bcrypt


class UserStore:
    """Simple JSON file-based user store."""

    def __init__(self, path: Path):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save({})

    def _load(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get_user(self, email: str) -> Optional[dict]:
        users = self._load()
        return users.get(email.lower())

    def create_user(self, email: str, password: str) -> bool:
        users = self._load()
        email = email.lower()
        if email in users:
            return False
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users[email] = {
            "email": email,
            "password_hash": hashed.decode(),
        }
        self._save(users)
        return True

    def verify_password(self, email: str, password: str) -> bool:
        user = self.get_user(email)
        if not user:
            return False
        return bcrypt.checkpw(password.encode(), user["password_hash"].encode())

    def update_password(self, email: str, new_password: str) -> bool:
        users = self._load()
        email = email.lower()
        if email not in users:
            return False
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        users[email]["password_hash"] = hashed.decode()
        self._save(users)
        return True

    def delete_user(self, email: str) -> bool:
        users = self._load()
        email = email.lower()
        if email not in users:
            return False
        del users[email]
        self._save(users)
        return True

    def list_users(self) -> list:
        users = self._load()
        return [{"email": u["email"]} for u in users.values()]

    def has_users(self) -> bool:
        return len(self._load()) > 0


def generate_secret_key() -> str:
    return secrets.token_hex(32)
