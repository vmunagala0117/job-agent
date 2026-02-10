"""Notification service for delivering job matches via email, Teams, or Slack."""

import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

from .models import NotificationChannel, NotificationConfig, RankedJob, UserProfile

logger = logging.getLogger(__name__)


class NotificationSender(ABC):
    """Abstract base class for notification senders."""
    
    @abstractmethod
    async def send(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Send a notification. Returns True if successful."""
        pass


class EmailSender(NotificationSender):
    """Send notifications via email."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Send email notification."""
        if not self.config.email_to or not self.config.smtp_host:
            logger.warning("Email not configured")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.smtp_user or "job-agent@localhost"
            msg["To"] = self.config.email_to
            
            # Add plain text part
            msg.attach(MIMEText(body, "plain"))
            
            # Add HTML part if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))
            
            # Send via SMTP
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {self.config.email_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class TeamsSender(NotificationSender):
    """Send notifications via Microsoft Teams webhook."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Send Teams notification via webhook."""
        if not self.config.teams_webhook_url:
            logger.warning("Teams webhook not configured")
            return False
        
        try:
            # Teams Adaptive Card format
            card = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": subject,
                "sections": [{
                    "activityTitle": subject,
                    "text": body.replace("\n", "<br>"),
                    "markdown": True,
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.teams_webhook_url,
                    json=card,
                    timeout=30.0,
                )
                response.raise_for_status()
            
            logger.info("Teams notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False


class SlackSender(NotificationSender):
    """Send notifications via Slack webhook."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Send Slack notification via webhook."""
        if not self.config.slack_webhook_url:
            logger.warning("Slack webhook not configured")
            return False
        
        try:
            payload = {
                "text": f"*{subject}*",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": subject}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": body}
                    }
                ]
            }
            
            if self.config.slack_channel:
                payload["channel"] = self.config.slack_channel
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
            
            logger.info("Slack notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class ConsoleSender(NotificationSender):
    """Print notifications to console (for testing/development)."""
    
    async def send(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Print notification to console."""
        print("\n" + "=" * 60)
        print(f"📬 NOTIFICATION: {subject}")
        print("=" * 60)
        print(body)
        print("=" * 60 + "\n")
        return True


class NotificationService:
    """Service for sending job match notifications."""
    
    def __init__(self, configs: list[NotificationConfig] = None):
        self.senders: list[NotificationSender] = []
        
        if configs:
            for config in configs:
                if not config.enabled:
                    continue
                    
                if config.channel == NotificationChannel.EMAIL:
                    self.senders.append(EmailSender(config))
                elif config.channel == NotificationChannel.TEAMS:
                    self.senders.append(TeamsSender(config))
                elif config.channel == NotificationChannel.SLACK:
                    self.senders.append(SlackSender(config))
                elif config.channel == NotificationChannel.CONSOLE:
                    self.senders.append(ConsoleSender())
        else:
            # Default to console if no config
            self.senders.append(ConsoleSender())
    
    async def send_job_matches(
        self,
        ranked_jobs: list[RankedJob],
        profile: UserProfile,
        title: str = "New Job Matches",
    ) -> dict[str, bool]:
        """Send notification with ranked job matches.
        
        Returns dict of {channel: success} for each configured sender.
        """
        if not ranked_jobs:
            return {}
        
        # Build notification content
        subject = f"🎯 {title} - {len(ranked_jobs)} matches for {profile.name}"
        
        body = self._format_job_matches_text(ranked_jobs, profile)
        html_body = self._format_job_matches_html(ranked_jobs, profile)
        
        # Send via all configured channels
        results = {}
        for sender in self.senders:
            channel_name = type(sender).__name__.replace("Sender", "").lower()
            results[channel_name] = await sender.send(subject, body, html_body)
        
        return results
    
    def _format_job_matches_text(self, ranked_jobs: list[RankedJob], profile: UserProfile) -> str:
        """Format job matches as plain text."""
        lines = [
            f"Hi {profile.name}!\n",
            f"Found {len(ranked_jobs)} job matches based on your profile.\n",
            "=" * 50,
        ]
        
        for i, rj in enumerate(ranked_jobs, 1):
            job = rj.job
            salary = ""
            if job.salary_min and job.salary_max:
                salary = f" | ${job.salary_min:,} - ${job.salary_max:,}"
            elif job.salary_min:
                salary = f" | ${job.salary_min:,}+"
            
            lines.append(f"\n#{i}. {job.title} at {job.company}")
            lines.append(f"   📍 {job.location}{salary}")
            lines.append(f"   🎯 Match Score: {rj.score * 100:.0f}%")
            lines.append(f"   📊 Similarity: {rj.similarity_score * 100:.0f}% | Skills: {rj.skill_match_score * 100:.0f}%")
            if job.url:
                lines.append(f"   🔗 {job.url}")
            lines.append(f"   💬 {rj.justification}")
            lines.append("")
            
            # Quick actions
            lines.append("   Quick Actions: [Good Fit] [Not Relevant] [Tailor Resume] [Draft Cover Letter]")
            lines.append("-" * 50)
        
        lines.append("\nReply with the job number and action to provide feedback.")
        return "\n".join(lines)
    
    def _format_job_matches_html(self, ranked_jobs: list[RankedJob], profile: UserProfile) -> str:
        """Format job matches as HTML for email."""
        html_parts = [
            f"<h2>Hi {profile.name}!</h2>",
            f"<p>Found <strong>{len(ranked_jobs)}</strong> job matches based on your profile.</p>",
            "<hr>",
        ]
        
        for i, rj in enumerate(ranked_jobs, 1):
            job = rj.job
            salary = ""
            if job.salary_min and job.salary_max:
                salary = f" | ${job.salary_min:,} - ${job.salary_max:,}"
            elif job.salary_min:
                salary = f" | ${job.salary_min:,}+"
            
            html_parts.append(f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
                <h3 style="margin: 0 0 10px 0;">#{i}. {job.title} at {job.company}</h3>
                <p style="margin: 5px 0;">📍 {job.location}{salary}</p>
                <p style="margin: 5px 0;">
                    <strong>🎯 Match Score: {rj.score * 100:.0f}%</strong>
                    <span style="color: #666;">
                        (Similarity: {rj.similarity_score * 100:.0f}% | Skills: {rj.skill_match_score * 100:.0f}%)
                    </span>
                </p>
                <p style="margin: 5px 0; font-style: italic;">{rj.justification}</p>
                {f'<p><a href="{job.url}" style="color: #0066cc;">View Job Posting →</a></p>' if job.url else ''}
                <div style="margin-top: 10px;">
                    <a href="#" style="padding: 5px 10px; background: #28a745; color: white; text-decoration: none; border-radius: 4px; margin-right: 5px;">✓ Good Fit</a>
                    <a href="#" style="padding: 5px 10px; background: #dc3545; color: white; text-decoration: none; border-radius: 4px; margin-right: 5px;">✗ Not Relevant</a>
                    <a href="#" style="padding: 5px 10px; background: #17a2b8; color: white; text-decoration: none; border-radius: 4px; margin-right: 5px;">📝 Tailor Resume</a>
                    <a href="#" style="padding: 5px 10px; background: #6c757d; color: white; text-decoration: none; border-radius: 4px;">✉️ Draft Cover Letter</a>
                </div>
            </div>
            """)
        
        html_parts.append("<p style='color: #666;'>Reply to this email or use the buttons above to provide feedback.</p>")
        
        return "\n".join(html_parts)


def get_notification_service(configs: list[NotificationConfig] = None) -> NotificationService:
    """Factory function to get notification service with configured channels."""
    import os
    
    if configs:
        return NotificationService(configs)
    
    # Auto-configure from environment
    auto_configs = []
    
    # Check for Teams webhook
    teams_url = os.getenv("TEAMS_WEBHOOK_URL")
    if teams_url:
        auto_configs.append(NotificationConfig(
            channel=NotificationChannel.TEAMS,
            teams_webhook_url=teams_url,
        ))
    
    # Check for Slack webhook
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_url:
        auto_configs.append(NotificationConfig(
            channel=NotificationChannel.SLACK,
            slack_webhook_url=slack_url,
            slack_channel=os.getenv("SLACK_CHANNEL"),
        ))
    
    # Check for email config
    smtp_host = os.getenv("SMTP_HOST")
    email_to = os.getenv("NOTIFICATION_EMAIL")
    if smtp_host and email_to:
        auto_configs.append(NotificationConfig(
            channel=NotificationChannel.EMAIL,
            email_to=email_to,
            smtp_host=smtp_host,
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
        ))
    
    # Always add console as fallback
    auto_configs.append(NotificationConfig(channel=NotificationChannel.CONSOLE))
    
    return NotificationService(auto_configs)
