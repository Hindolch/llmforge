import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

def send_email_alert(alert_type: str):
    SMTP_SERVER = "smtp-relay.brevo.com"
    SMTP_PORT = 587
    SMTP_USERNAME = os.getenv("BREVO_SMTP_USER")
    SMTP_PASSWORD = os.getenv("BREVO_SMTP_PASS")

    sender_email = "roychoudhuryhindol@gmail.com"  # Must be verified with Brevo
    receiver_email = "roychoudhuryhindol@gmail.com"

    subject = f"⚠️ {alert_type} - LLMForge Pipeline Triggered"
    body = f"The LLMForge pipeline just started due to: {alert_type}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("📧 Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
