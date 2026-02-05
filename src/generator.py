"""
Response Generator

Uses Claude to generate copy feedback in Stefan's voice.
"""

from typing import Optional, List, Dict

import anthropic

from . import config
from .retriever import get_retriever


class CopyChiefGenerator:
    """Generates copy feedback using Claude with RAG context."""

    def __init__(self):
        """Initialize the generator with Anthropic client."""
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")

        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.retriever = get_retriever()

    def generate_feedback(
        self,
        copy_text: str,
        copy_type: Optional[str] = None,
        additional_context: Optional[str] = None,
        use_rag: bool = True,
        top_k: int = None,
        word_count: int = None,
    ) -> str:
        """
        Generate copy feedback in Stefan's voice.

        Args:
            copy_text: The copy to review
            copy_type: Type of copy (PDP, VSL, email, etc.)
            additional_context: User-provided context about the copy
            use_rag: Whether to retrieve context from past reviews
            top_k: Number of context chunks to retrieve

        Returns:
            Generated feedback text
        """
        # Get context from past reviews
        context = ""
        if use_rag and self.retriever.is_ready:
            results = self.retriever.retrieve_diverse(
                copy_text,
                top_k=top_k or config.TOP_K_RESULTS
            )
            context = self.retriever.format_context(results)

        # Build the system prompt with context
        system_prompt = config.STEFAN_SYSTEM_PROMPT.format(context=context)

        # Build the user message with context
        user_message_parts = []

        # Include actual word count so model doesn't have to estimate
        if word_count:
            user_message_parts.append(f"## Copy Statistics\n**Actual word count: {word_count:,} words** (use this exact number in your feedback, do not estimate or round)\n")

        if additional_context:
            user_message_parts.append(f"## Context\n{additional_context}\n")

        user_message_parts.append(f"## Copy to Review\n\n{copy_text}")

        user_message = "\n".join(user_message_parts)

        # Call Claude
        message = self.client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return message.content[0].text

    def generate_feedback_streaming(
        self,
        copy_text: str,
        copy_type: Optional[str] = None,
        additional_context: Optional[str] = None,
        use_rag: bool = True,
        top_k: int = None,
        word_count: int = None,
    ):
        """
        Generate copy feedback with streaming response.

        Yields text chunks as they're generated.
        """
        # Get context from past reviews
        context = ""
        if use_rag and self.retriever.is_ready:
            results = self.retriever.retrieve_diverse(
                copy_text,
                top_k=top_k or config.TOP_K_RESULTS
            )
            context = self.retriever.format_context(results)

        # Build the system prompt with context
        system_prompt = config.STEFAN_SYSTEM_PROMPT.format(context=context)

        # Build the user message with context
        user_message_parts = []

        # Include actual word count so model doesn't have to estimate
        if word_count:
            user_message_parts.append(f"## Copy Statistics\n**Actual word count: {word_count:,} words** (use this exact number in your feedback, do not estimate or round)\n")

        if additional_context:
            user_message_parts.append(f"## Context\n{additional_context}\n")

        user_message_parts.append(f"## Copy to Review\n\n{copy_text}")

        user_message = "\n".join(user_message_parts)

        # Call Claude with streaming
        with self.client.messages.stream(
            model=config.LLM_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        ) as stream:
            for text in stream.text_stream:
                yield text


class CopyChiefChatGenerator:
    """Handles follow-up chat conversations after initial review."""

    def __init__(self):
        """Initialize the chat generator."""
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")

        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.retriever = get_retriever()

    def chat_response(
        self,
        messages: List[Dict],
        original_copy: str,
        use_rag: bool = True,
        top_k: int = None,
    ) -> str:
        """
        Generate a response to a follow-up message.

        Args:
            messages: Conversation history (list of {"role": str, "content": str})
            original_copy: The original copy being discussed
            use_rag: Whether to use RAG context
            top_k: Number of context chunks

        Returns:
            Response text
        """
        # Get RAG context based on the latest user message
        context = ""
        if use_rag and self.retriever.is_ready:
            # Use the last user message for retrieval
            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                ""
            )
            if last_user_msg:
                results = self.retriever.retrieve_diverse(
                    last_user_msg + " " + original_copy[:500],
                    top_k=top_k or config.TOP_K_RESULTS
                )
                context = self.retriever.format_context(results)

        # Determine how much of original copy to include based on length
        # For very long copy, include beginning and end to give context
        if len(original_copy) > 15000:
            copy_preview = f"{original_copy[:8000]}\n\n[... middle section omitted for context length - full copy is {len(original_copy):,} characters ...]\n\n{original_copy[-4000:]}"
        elif len(original_copy) > 8000:
            copy_preview = f"{original_copy[:6000]}\n\n[... {len(original_copy) - 6000:,} more characters ...]\n\n{original_copy[-2000:]}"
        else:
            copy_preview = original_copy

        # Build system prompt for chat
        chat_system_prompt = f"""You are Stefan, continuing a copy review conversation.

You've already provided initial feedback on a piece of copy. Now the user is asking follow-up questions or requesting rewrites.

## YOUR STYLE
- Stay in character as Stefan (direct, blunt, actionable)
- If asked for a rewrite, actually provide the rewritten copy
- Be helpful and specific
- Reference your earlier feedback when relevant

## IMPORTANT: REWRITE GUIDANCE
When asked to rewrite LONG copy (VSLs, long sales letters):
- If the original is 5,000+ words, do NOT try to rewrite the entire thing in one response
- Instead, rewrite section by section: "Here's the hook/lead rewritten... here's the mechanism section... etc."
- Or ask which specific section they want rewritten first
- For shorter copy (ads, emails, landing pages), full rewrites are fine

## ORIGINAL COPY BEING DISCUSSED
{copy_preview}

## RELEVANT EXAMPLES FROM PAST REVIEWS
{context}

Respond naturally as Stefan would in a conversation."""

        # Build message history for Claude
        claude_messages = []
        for msg in messages:
            claude_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Call Claude with high token limit for rewrites
        message = self.client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=15000,
            system=chat_system_prompt,
            messages=claude_messages
        )

        return message.content[0].text


# Singleton instances
_generator = None
_chat_generator = None


def get_generator() -> CopyChiefGenerator:
    """Get or create the generator singleton."""
    global _generator
    if _generator is None:
        _generator = CopyChiefGenerator()
    return _generator


def get_chat_generator() -> CopyChiefChatGenerator:
    """Get or create the chat generator singleton."""
    global _chat_generator
    if _chat_generator is None:
        _chat_generator = CopyChiefChatGenerator()
    return _chat_generator
