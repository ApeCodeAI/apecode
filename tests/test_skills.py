"""Tests for skill discovery."""

from __future__ import annotations

from pathlib import Path

from apecode.skills import Skill, SkillCatalog


def test_discover_nested_skill_dirs(tmp_path: Path) -> None:
    (tmp_path / "skills" / "one").mkdir(parents=True)
    (tmp_path / "skills" / "two").mkdir(parents=True)
    (tmp_path / "skills" / "one" / "SKILL.md").write_text(
        "# One\n\nFirst skill",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "two" / "SKILL.md").write_text(
        "# Two\n\nSecond skill",
        encoding="utf-8",
    )

    catalog = SkillCatalog.from_roots([tmp_path / "skills"])
    names = [skill.name for skill in catalog.list_skills()]
    assert names == ["one", "two"]
    assert "First skill" in catalog.format_overview()
    assert "### How to use skills" in catalog.format_for_system_prompt()


def test_merge_inline_skill() -> None:
    catalog = SkillCatalog.from_roots([])
    merged = catalog.with_additional(
        [
            Skill(
                name="Inline",
                description="Inline plugin skill",
                inline_content="# Inline\n\nUse concise output.",
                source="plugin:test",
            )
        ]
    )
    skill = merged.get("inline")
    assert skill is not None
    assert "Use concise output." in skill.read_text()
