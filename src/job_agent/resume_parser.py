"""Resume parsing and skill extraction from PDF/DOCX files."""

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ParsedResume:
    """Structured data extracted from a resume."""
    
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    current_title: str = ""
    years_experience: Optional[int] = None
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    
    def to_user_profile(self, profile_id: Optional[str] = None) -> UserProfile:
        """Convert parsed resume to a UserProfile."""
        profile = UserProfile(
            name=self.name,
            email=self.email,
            resume_text=self.raw_text,
            summary=self.summary,
            skills=self.skills,
            years_experience=self.years_experience,
            current_title=self.current_title,
        )
        if profile_id:
            profile.id = profile_id
        return profile


class ResumeParser:
    """Parses resume files (PDF, DOCX) and extracts structured information."""
    
    def __init__(self, llm_client=None):
        """Initialize parser with optional LLM client for smart extraction.
        
        Args:
            llm_client: Azure OpenAI chat client for LLM-based extraction.
                       If None, uses regex-based extraction only.
        """
        self.llm_client = llm_client
    
    def parse_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n".join(text_parts)
    
    def parse_pdf_bytes(self, data: bytes) -> str:
        """Extract text from PDF bytes."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=data, filetype="pdf")
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n".join(text_parts)
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        from docx import Document
        
        doc = Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        return "\n".join(text_parts)
    
    def parse_docx_bytes(self, data: bytes) -> str:
        """Extract text from DOCX bytes."""
        from docx import Document
        
        doc = Document(io.BytesIO(data))
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        return "\n".join(text_parts)
    
    def parse_file(self, file_path: str) -> str:
        """Extract text from a file based on extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            return self.parse_pdf(file_path)
        elif extension in (".docx", ".doc"):
            return self.parse_docx(file_path)
        elif extension == ".txt":
            return path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def parse_base64(self, data: str, file_type: str) -> str:
        """Parse base64-encoded file data.
        
        Args:
            data: Base64-encoded file content
            file_type: File type (pdf, docx, txt)
        """
        decoded = base64.b64decode(data)
        
        if file_type.lower() == "pdf":
            return self.parse_pdf_bytes(decoded)
        elif file_type.lower() in ("docx", "doc"):
            return self.parse_docx_bytes(decoded)
        elif file_type.lower() == "txt":
            return decoded.decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def extract_with_regex(self, text: str) -> ParsedResume:
        """Extract structured data using regex patterns.
        
        This is a fallback when LLM is not available.
        """
        parsed = ParsedResume(raw_text=text)
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            parsed.email = email_match.group()
        
        # Extract phone
        phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            parsed.phone = phone_match.group()
        
        # Common skills to look for
        common_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "Git", "CI/CD", "Jenkins", "GitHub Actions",
            "Agile", "Scrum", "Leadership", "Project Management",
            "API", "REST", "GraphQL", "Microservices",
        ]
        
        # Find skills mentioned in text
        text_lower = text.lower()
        for skill in common_skills:
            if skill.lower() in text_lower:
                parsed.skills.append(skill)
        
        # Try to extract years of experience
        exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)'
        exp_match = re.search(exp_pattern, text, re.IGNORECASE)
        if exp_match:
            parsed.years_experience = int(exp_match.group(1))
        
        return parsed
    
    async def extract_with_llm(self, text: str) -> ParsedResume:
        """Extract structured data using LLM.
        
        Uses Azure OpenAI to intelligently parse the resume.
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured")
        
        prompt = """Analyze this resume and extract the following information in JSON format:

{
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "summary": "2-3 sentence professional summary",
    "current_title": "Current or most recent job title",
    "years_experience": null or integer,
    "skills": ["List", "of", "technical", "and", "soft", "skills"],
    "education": ["Degree, University, Year"],
    "certifications": ["Certification names"]
}

Resume text:
---
""" + text[:8000] + """
---

Return only valid JSON, no additional text."""

        try:
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            
            # Parse the JSON response
            content = response.content
            # Clean up common issues
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content.strip())
            
            return ParsedResume(
                raw_text=text,
                name=data.get("name", ""),
                email=data.get("email", ""),
                phone=data.get("phone", ""),
                summary=data.get("summary", ""),
                current_title=data.get("current_title", ""),
                years_experience=data.get("years_experience"),
                skills=data.get("skills", []),
                education=data.get("education", []),
                certifications=data.get("certifications", []),
            )
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to regex")
            return self.extract_with_regex(text)
    
    async def parse_and_extract(
        self,
        file_path: Optional[str] = None,
        file_data: Optional[str] = None,
        file_type: Optional[str] = None,
        use_llm: bool = True,
    ) -> ParsedResume:
        """Parse a resume file and extract structured data.
        
        Args:
            file_path: Path to the resume file
            file_data: Base64-encoded file content (alternative to file_path)
            file_type: File type when using file_data (pdf, docx, txt)
            use_llm: Whether to use LLM for extraction (falls back to regex if fails)
        
        Returns:
            ParsedResume with extracted information
        """
        # Extract text from file
        if file_path:
            text = self.parse_file(file_path)
        elif file_data and file_type:
            text = self.parse_base64(file_data, file_type)
        else:
            raise ValueError("Must provide either file_path or (file_data and file_type)")
        
        # Extract structured data
        if use_llm and self.llm_client:
            return await self.extract_with_llm(text)
        else:
            return self.extract_with_regex(text)


def create_parser(llm_client=None) -> ResumeParser:
    """Factory function to create a resume parser."""
    return ResumeParser(llm_client)
