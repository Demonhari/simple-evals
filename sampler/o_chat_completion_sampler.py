import time
from typing import Any
import os
import openai
from openai import OpenAI

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class OChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(
        self,
        *,
        reasoning_effort: str | None = None,
        model: str = "o1-mini",
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort

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
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    reasoning_effort=self.reasoning_effort,
                )
                content = response.choices[0].message.content
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
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

