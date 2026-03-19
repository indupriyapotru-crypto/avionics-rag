"""
summarizer.py – Generate structured avionics summaries via Groq / LLaMA.
"""

import logging

from groq import Groq, GroqError
from config import GROQ_API_KEY, GROQ_MODEL

log    = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

_PROMPT_TEMPLATE = """\
You are an avionics news editor. The user wants to know about: "{topic}"

Read the articles below and write a clean, flowing digest focused on that topic.

Rules:
- Plain prose only — no headers, no bullet points, no "Sources:" section
- 4-6 sentences, covering the most relevant developments to the topic
- Do not repeat the same fact twice
- Skip filler phrases like "In summary" or "It is worth noting"
- If the articles don't directly cover the topic, summarise what is most
  related and note that no exact match was found today

NEWS ARTICLES:
{context}
"""


def summarize(docs: list[dict], topic: str = "latest avionics news", model: str = GROQ_MODEL) -> str:
    """
    Summarise retrieved chunks, focused on `topic`.

    Parameters
    ----------
    docs  : chunks returned by VectorStore.search()
    topic : the user's original query — keeps the summary on-topic
    model : Groq model string (overridable for testing)

    Returns
    -------
    Plain prose string (no Sources block — those are rendered by the UI).
    """
    if not docs:
        return "⚠️ No relevant news found to summarise."

    context = "\n\n".join(
        f"[{i+1}] {d['title']} ({d.get('published', '')})\n{d['text']}"
        for i, d in enumerate(docs)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(
                topic=topic, context=context
            )}],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    except GroqError as exc:
        log.error("Groq API error: %s", exc)
        return f"⚠️ Summary generation failed: {exc}"
