"""Wiki CRUD over markdown files for the eval harness.

Each WikiDB instance owns an isolated directory:
    <wiki_dir>/
        index.md          — catalog: one line per atom
        log.md            — append-only event log
        atoms/
            <atom_id>.md  — atom file with YAML frontmatter + content

Passing a WikiDB instance through the call chain makes concurrent
per-question wikis safe — no shared global state.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


class WikiDB:
    def __init__(self, wiki_dir: Path) -> None:
        self.wiki_dir = Path(wiki_dir)
        self.index_file = self.wiki_dir / "index.md"
        self.log_file = self.wiki_dir / "log.md"
        self.atoms_dir = self.wiki_dir / "atoms"

    def _ensure(self) -> None:
        self.atoms_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self.index_file.write_text("# Index\n\n")
        if not self.log_file.exists():
            self.log_file.write_text("# Log\n\n")

    @property
    def _subjects_file(self) -> Path:
        return self.wiki_dir / "subjects.json"

    @property
    def _word_index_file(self) -> Path:
        return self.wiki_dir / "word_index.json"

    def _load_subjects(self) -> dict[str, str]:
        return json.loads(self._subjects_file.read_text()) if self._subjects_file.exists() else {}

    def register_subject(self, subject: str, atom_id: str) -> str | None:
        """Register subject → atom_id. Returns the displaced atom_id if the subject existed."""
        key = subject.lower().strip()
        subjects = self._load_subjects()
        old_id = subjects.get(key)
        subjects[key] = atom_id
        self._subjects_file.write_text(json.dumps(subjects))
        return old_id if old_id != atom_id else None

    def lookup_subject(self, subject: str) -> str | None:
        return self._load_subjects().get(subject.lower().strip())

    def update_word_index(self, atom_id: str, content: str) -> None:
        """Add atom_id to each content word's posting list in word_index.json."""
        words = set(re.findall(r"[a-z0-9]{3,}", content.lower()))
        index: dict[str, list[str]] = (
            json.loads(self._word_index_file.read_text())
            if self._word_index_file.exists()
            else {}
        )
        for word in words:
            ids = index.get(word, [])
            if atom_id not in ids:
                ids.append(atom_id)
            index[word] = ids
        self._word_index_file.write_text(json.dumps(index))

    def load_word_index(self) -> dict[str, list[str]]:
        return json.loads(self._word_index_file.read_text()) if self._word_index_file.exists() else {}

    def reset(self) -> None:
        """Delete all atoms, reset index.md, subjects.json, word_index.json, and write fresh log.md."""
        if self.atoms_dir.exists():
            shutil.rmtree(self.atoms_dir)
        self.atoms_dir.mkdir(parents=True)
        self.index_file.write_text("# Index\n\n")
        if self._subjects_file.exists():
            self._subjects_file.unlink()
        if self._word_index_file.exists():
            self._word_index_file.unlink()
        ts = datetime.now(tz=timezone.utc).isoformat()
        self.log_file.write_text(f"# Log\n\n## [{ts}] reset\n")

    def teardown(self) -> None:
        """Remove the entire wiki directory after a question is processed."""
        if self.wiki_dir.exists():
            shutil.rmtree(self.wiki_dir)

    def write_atom(
        self,
        atom_id: str,
        content: str,
        kind: str,
        valid_from: datetime,
        subject: str = "",
        supersedes: str | None = None,
    ) -> None:
        self._ensure()
        text = (
            f"---\n"
            f"atom_id: {atom_id}\n"
            f"kind: {kind}\n"
            f"subject: {subject}\n"
            f"valid_from: {valid_from.isoformat()}\n"
            f"valid_until: null\n"
            f"is_superseded: false\n"
            f"superseded_by: null\n"
            f"supersedes: {supersedes or 'null'}\n"
            f"---\n\n"
            f"{content}\n"
        )
        (self.atoms_dir / f"{atom_id}.md").write_text(text)

    def mark_superseded(self, atom_id: str, superseded_by: str) -> None:
        path = self.atoms_dir / f"{atom_id}.md"
        if not path.exists():
            return
        text = path.read_text()
        text = re.sub(r"^is_superseded: false$", "is_superseded: true", text, flags=re.MULTILINE)
        text = re.sub(r"^superseded_by: null$", f"superseded_by: {superseded_by}", text, flags=re.MULTILINE)
        path.write_text(text)

    def update_index(self, atom_id: str, summary: str, kind: str, valid_from: datetime, subject: str = "") -> None:
        self._ensure()
        subject_part = f" | {subject}" if subject else ""
        line = f"- [{atom_id}](atoms/{atom_id}.md) | {kind} | {valid_from.date()}{subject_part} | {summary}\n"
        with self.index_file.open("a") as f:
            f.write(line)

    def list_subjects(self) -> dict[str, str]:
        """Return current subject → atom_id mapping (all subjects, including superseded targets)."""
        return self._load_subjects()

    def read_index(self) -> str:
        self._ensure()
        return self.index_file.read_text()

    def read_active_index(self) -> str:
        """Return index content with superseded atoms excluded.

        Reads each atom file to check is_superseded — keeps the index as the
        single source of atom ordering while filtering stale entries.
        """
        self._ensure()
        raw = self.index_file.read_text()
        active_lines: list[str] = []
        for line in raw.splitlines(keepends=True):
            if not line.startswith("- ["):
                active_lines.append(line)
                continue
            try:
                atom_id = line[3 : line.index("]")]
            except ValueError:
                active_lines.append(line)
                continue
            if "is_superseded: true" not in self.read_atom(atom_id):
                active_lines.append(line)
        return "".join(active_lines)

    def read_atom(self, atom_id: str) -> str:
        path = self.atoms_dir / f"{atom_id}.md"
        return path.read_text() if path.exists() else ""

    def list_atoms(self) -> list[str]:
        self._ensure()
        return [p.stem for p in sorted(self.atoms_dir.glob("*.md"))]

    def append_log(self, event: str, detail: str = "") -> None:
        self._ensure()
        ts = datetime.now(tz=timezone.utc).isoformat()
        with self.log_file.open("a") as f:
            f.write(f"\n## [{ts}] {event}\n")
            if detail:
                f.write(f"{detail}\n")
