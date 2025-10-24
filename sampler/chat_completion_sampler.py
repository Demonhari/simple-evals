import time
from typing import Any

import openai
from openai import OpenAI
import os
from ..eval_types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                # build kwargs dynamically
                kwargs = {
                    "model": self.model,
                    "messages": message_list,
                }

                # GPT-3.5, GPT-4-turbo, and GPT-4.1 allow custom temperature
                if any(x in self.model for x in ["gpt-3.5", "gpt-4.1", "gpt-4-turbo"]):
                    kwargs["temperature"] = self.temperature

                # newer models (GPT-4o, GPT-5, etc.) use `max_completion_tokens`
                if any(x in self.model for x in ["gpt-4o", "gpt-5"]):
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["max_tokens"] = self.max_tokens

                response = self.client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )

            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg:
                    wait_time = min(30, 5 * (2 ** trial))  # adaptive backoff, max 30s
                    if int(os.getenv("HB_DEBUG", "0")):
                        print(f"[Sampler pacing] ⚠️ 429 rate limit hit. Backing off {exception_backoff:.1f}s (trial {trial})")
                    time.sleep(wait_time)
                    trial += 1
                    continue
                elif "BadRequestError" in msg:
                    print(f"[Sampler pacing] BadRequestError encountered — skipping sample.")
                    return SamplerResponse(
                        response_text="No response (bad request).",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )
                else:
                    print(f"[Sampler pacing] Unexpected error: {msg}")
                    raise


