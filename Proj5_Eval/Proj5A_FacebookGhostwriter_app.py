"""Facebook post ghostwriter v1A using OpenAI API.

Generates an inspiring post about MacKenzie Scott's philanthropy
in a 3-part format: Hook, Body (150 words), and extended CTA (100 words).
"""

import logging
import os
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.8
MAX_TOKENS = 800

ARTICLE_URLS = [
    "https://finance.yahoo.com/news/mackenzie-scott-says-her-college-133042714.html",
    "https://apnews.com/article/mackenzie-scott-princeton-university-funding-u-ce25dd95df46653c13d0701c49c98c51",
]

SYSTEM_PROMPT = (
    "You are a Facebook post ghostwriter specializing in inspiring, "
    "authentic non-fiction stories. Write a single Facebook post using "
    "this exact 3-part structure:\n\n"
    "1. HOOK (1-2 lines): A compelling opening that stops the scroll. "
    "Use a surprising fact or emotional contrast.\n"
    "2. BODY (up to 150 words): Deliver the story concisely with only "
    "the most impactful details — names, numbers, and turning points. "
    "Keep paragraphs short for mobile readability.\n"
    "3. CTA (up to 100 words): A longer, persuasive call to action "
    "grounded in reflection on the stories. Connect the reader's own "
    "life to the themes of generosity and opportunity. Invite them to "
    "consider how small acts of kindness ripple outward.\n\n"
    "Tone: Warm, genuine, and uplifting — never preachy or clickbaity.\n"
    "Format: Use line breaks between sections. No hashtags. No emojis."
)


@dataclass
class ArticleContext:
    """Structured container for article facts used in post generation."""

    roommate_name: str
    loan_amount: str
    college: str
    funding_u_total: str
    students_helped: str
    scott_contribution_ratio: str
    scott_total_donated: str
    article_urls: list[str] = field(default_factory=list)


def load_api_key() -> str:
    """Load and validate OpenAI API key from environment."""
    load_dotenv("OpenAI_key.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in OpenAI_key.env")
    return api_key


def build_article_context() -> ArticleContext:
    """Return structured facts from the MacKenzie Scott articles."""
    return ArticleContext(
        roommate_name="Jeannie Tarkenton",
        loan_amount="$1,000",
        college="Princeton",
        funding_u_total="$80 million",
        students_helped="approximately 8,000",
        scott_contribution_ratio="30 cents for every dollar loaned",
        scott_total_donated="$26 billion",
        article_urls=ARTICLE_URLS,
    )


def build_user_message(context: ArticleContext) -> str:
    """Format article facts into a user prompt for the LLM."""
    return (
        f"Write an inspiring Facebook post based on these facts about "
        f"MacKenzie Scott:\n\n"
        f"- As a sophomore at {context.college}, Scott was about to drop out "
        f"when her roommate {context.roommate_name} found her crying and "
        f"loaned her {context.loan_amount} to stay in school.\n"
        f"- Years later, {context.roommate_name} founded Funding U, which has "
        f"provided {context.funding_u_total} in low-interest loans to "
        f"{context.students_helped} students.\n"
        f"- Scott now provides most of the junior debt Funding U uses, "
        f"contributing {context.scott_contribution_ratio}.\n"
        f"- Scott has donated {context.scott_total_donated} total.\n\n"
        f"Sources:\n"
        f"- {context.article_urls[0]}\n"
        f"- {context.article_urls[1]}"
    )


def generate_post(api_key: str, context: ArticleContext) -> str:
    """Call OpenAI API to generate the Facebook post."""
    client = openai.OpenAI(api_key=api_key)
    user_message = build_user_message(context)

    logger.info("Generating Facebook post via %s...", MODEL_NAME)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except openai.APIError as exc:
        logger.error("OpenAI API request failed: %s", exc)
        raise

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response")
    return content


def main() -> None:
    """Orchestrate post generation: load config, build context, generate."""
    api_key = load_api_key()
    context = build_article_context()
    post = generate_post(api_key, context)

    print("\n" + "=" * 60)
    print("GENERATED FACEBOOK POST")
    print("=" * 60 + "\n")
    print(post)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
