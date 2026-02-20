"""Skill discovery and rendering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True, slots=True)
class Skill:
    """One discovered skill."""

    name: str
    description: str
    path: Path | None = None
    inline_content: str | None = None
    source: str = "local"

    def read_text(self) -> str:
        """Read the skill body."""
        if self.inline_content is not None:
            return self.inline_content.strip()
        if self.path is None:
            return ""
        return self.path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_description(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        return line[:160]
    return "No description."


def _iter_skill_files(roots: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    for root in roots:
        normalized = root.expanduser().resolve()
        if not normalized.exists() or not normalized.is_dir():
            continue

        direct = normalized / SKILL_FILE_NAME
        if direct.exists() and direct.is_file():
            discovered.append(direct)

        for child in sorted(normalized.iterdir()):
            if not child.is_dir():
                continue
            nested = child / SKILL_FILE_NAME
            if nested.exists() and nested.is_file():
                discovered.append(nested)
    return discovered


@dataclass(slots=True)
class SkillCatalog:
    """In-memory skill index."""

    _skills: dict[str, Skill]

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower().replace(" ", "-")

    @classmethod
    def from_roots(cls, roots: Iterable[Path]) -> SkillCatalog:
        indexed: dict[str, Skill] = {}
        for path in _iter_skill_files(roots):
            if path.parent.name.lower() == ".system":
                continue
            name = cls._normalize_name(path.parent.name)
            if name in indexed:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            indexed[name] = Skill(
                name=name,
                description=_extract_description(content),
                path=path,
                source=f"file:{path}",
            )
        return cls(_skills=indexed)

    def with_additional(self, skills: Iterable[Skill]) -> SkillCatalog:
        """Return a new catalog with additional skills merged by name."""
        indexed = dict(self._skills)
        for skill in skills:
            normalized = self._normalize_name(skill.name)
            if not normalized or normalized in indexed:
                continue
            indexed[normalized] = Skill(
                name=normalized,
                description=skill.description,
                path=skill.path,
                inline_content=skill.inline_content,
                source=skill.source,
            )
        return SkillCatalog(_skills=indexed)

    def list_skills(self) -> list[Skill]:
        """Return all discovered skills sorted by name."""
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        """Resolve skill by exact lowercase name."""
        return self._skills.get(self._normalize_name(name))

    def format_overview(self) -> str:
        """Render a compact skill summary."""
        skills = self.list_skills()
        if not skills:
            return "(none)"
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)

    def format_for_system_prompt(self) -> str:
        """Render a richer skills section inspired by production agents."""
        skills = self.list_skills()
        if not skills:
            return "(none)"
        lines = [
            "A skill is a local instruction bundle stored in `SKILL.md`.",
            "### Available skills",
        ]
        for skill in skills:
            location = str(skill.path) if skill.path is not None else skill.source
            lines.append(f"- {skill.name}: {skill.description} (source: {location})")
        lines.extend(
            [
                "### How to use skills",
                "- If the user names a skill explicitly, use it in this turn.",
                "- Read only the needed part of `SKILL.md` to keep context small.",
                "- Resolve relative references from the skill directory first.",
                "- If a skill cannot be loaded, explain briefly and fallback.",
            ]
        )
        return "\n".join(lines)
