"""Utilities to render application package artifacts and assemble ZIP archives.

Generates TXT/MD content and optional DOCX if python-docx is installed.
Produces an in-memory ZIP (bytes) suitable for streaming via FastAPI.
"""

from __future__ import annotations

import io
import zipfile
import datetime
import re
from typing import Iterable, List

def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _\-\.]+", "_", name)[:200]


def render_cover_letter_txt(pkg) -> str:
    return pkg.cover_letter or ""


def render_intro_email_txt(pkg) -> str:
    return pkg.intro_email or ""


def render_resume_suggestions_txt(pkg) -> str:
    if not pkg.resume_suggestions:
        return ""
    return "\n".join([f"- {s}" for s in pkg.resume_suggestions])


def render_job_description_txt(pkg) -> str:
    try:
        return pkg.job.description or ""
    except Exception:
        return ""


def render_cover_letter_md(pkg) -> str:
    # Simple markdown wrapper
    title = pkg.job.title if getattr(pkg.job, "title", None) else "Cover Letter"
    header = f"# Cover Letter — {title}\n\n"
    return header + (pkg.cover_letter or "")


def _render_docx_bytes_from_text(text: str, title: str = "Document") -> bytes | None:
    try:
        from docx import Document
    except Exception:
        return None

    doc = Document()
    doc.add_heading(title, level=1)
    for line in text.splitlines():
        if line.strip() == "":
            doc.add_paragraph("")
        else:
            doc.add_paragraph(line)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()


def packages_to_zip_bytes(packages: Iterable, formats: List[str] | None = None) -> bytes:
    """Return ZIP bytes containing artifacts for given packages.

    formats: list containing any of 'txt', 'md', 'docx'. Default ['txt','md']
    """
    if formats is None:
        formats = ["txt", "md"]

    buf = io.BytesIO()
    errors = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pkg in packages:
            try:
                # Create folder name per package
                job_title = getattr(pkg.job, "title", "job") or "job"
                folder = _sanitize_filename(f"{job_title}_{pkg.id[:8]}")

                # Cover letter
                cover_txt = render_cover_letter_txt(pkg)
                if "txt" in formats:
                    zf.writestr(f"{folder}/cover_letter.txt", cover_txt)
                if "md" in formats:
                    zf.writestr(f"{folder}/cover_letter.md", render_cover_letter_md(pkg))
                if "docx" in formats:
                    docb = _render_docx_bytes_from_text(cover_txt, title="Cover Letter")
                    if docb:
                        zf.writestr(f"{folder}/cover_letter.docx", docb)

                # Intro email
                intro_txt = render_intro_email_txt(pkg)
                if "txt" in formats:
                    zf.writestr(f"{folder}/intro_email.txt", intro_txt)
                if "docx" in formats:
                    docb = _render_docx_bytes_from_text(intro_txt, title="Intro Email")
                    if docb:
                        zf.writestr(f"{folder}/intro_email.docx", docb)

                # Resume suggestions
                rs_txt = render_resume_suggestions_txt(pkg)
                if rs_txt:
                    zf.writestr(f"{folder}/resume_suggestions.txt", rs_txt)

                # Job description
                jd = render_job_description_txt(pkg)
                if jd:
                    zf.writestr(f"{folder}/job_description.txt", jd)

                # Metadata file: include additional job attributes when present
                job = getattr(pkg, "job", None)
                profile_name = getattr(getattr(pkg, "profile", None), "name", "")
                job_title = getattr(job, "title", "") if job is not None else ""
                company = (getattr(job, "company", "") or getattr(job, "company_name", "")) if job is not None else ""
                location = getattr(job, "location", "") if job is not None else ""

                # Posting date: support datetime-like or string values
                posting_date_val = None
                if job is not None:
                    posting_date_val = getattr(job, "posted_at", None) or getattr(job, "posting_date", None) or getattr(job, "posted_date", None)
                posting_date = ""
                if posting_date_val is not None:
                    try:
                        posting_date = posting_date_val.isoformat()
                    except Exception:
                        posting_date = str(posting_date_val)

                # Salary and job URL (try a few common attribute names)
                salary = ""
                if job is not None:
                    salary = getattr(job, "salary_range", "") or getattr(job, "salary", "") or getattr(job, "compensation", "")
                job_url = ""
                if job is not None:
                    job_url = getattr(job, "url", "") or getattr(job, "job_url", "") or getattr(job, "link", "")

                meta_lines = [
                    f"id: {pkg.id}",
                    f"job_title: {job_title}",
                    f"company: {company}",
                    f"location: {location}",
                    f"profile: {profile_name}",
                    f"created_at: {getattr(pkg, 'created_at', '')}",
                    f"posting_date: {posting_date}",
                    f"salary: {salary}",
                    f"job_url: {job_url}",
                ]
                zf.writestr(f"{folder}/metadata.txt", "\n".join(meta_lines))
            except Exception as exc:
                # Record the failure and continue with other packages
                pkg_id = getattr(pkg, "id", "<unknown>")
                errors.append(f"Package {pkg_id} failed: {exc}")

        # If there were errors, add a top-level failures.txt describing them
        if errors:
            zf.writestr("FAILURES.txt", "\n".join(errors))

    buf.seek(0)
    return buf.read()
