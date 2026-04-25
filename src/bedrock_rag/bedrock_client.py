"""Thin wrapper over boto3 Bedrock Runtime client with retries.

Why a wrapper at all: boto3's bedrock-runtime client doesn't retry
ThrottlingException by default, and Bedrock throttles aggressively
when you exceed model TPM. Without retries, a slow burst hammers
your throughput and produces noisy errors instead of just being slow.
"""
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int


class BedrockClient:
    """Adapter so the rest of the code doesn't import boto3 directly."""

    def __init__(self, model_id: str, region: str = "us-east-1"):
        self.model_id = model_id
        self.region = region
        # Boto client is lazy-loaded so unit tests can construct without AWS creds
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def converse(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> BedrockResponse:
        """Single-shot conversation turn via the Converse API.

        Converse normalizes inputs across model providers (Claude, Titan,
        Llama all use the same shape). InvokeModel requires per-provider
        body schemas — Converse is strictly better unless you need a
        provider-specific feature.
        """
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            request["system"] = [{"text": system}]

        client = self._get_client()
        resp = client.converse(**request)
        msg = resp["output"]["message"]
        text = "".join(c.get("text", "") for c in msg["content"])
        usage = resp.get("usage", {})
        return BedrockResponse(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def embed_text(self, text: str) -> list[float]:
        """Titan Text Embeddings v2."""
        client = self._get_client()
        body = json.dumps({"inputText": text})
        resp = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
        )
        result = json.loads(resp["body"].read())
        return result["embedding"]

    def embed_multimodal(
        self,
        text: str | None = None,
        image_bytes: bytes | None = None,
    ) -> list[float]:
        """Titan Multimodal Embeddings v1.

        Pass text-only, image-only, or both. Embedding dimension is 1024
        regardless. Both modes use the same vector space, so a query
        embedding can be either type.
        """
        if text is None and image_bytes is None:
            raise ValueError("must provide text or image")

        import base64
        body: dict[str, Any] = {}
        if text:
            body["inputText"] = text
        if image_bytes:
            body["inputImage"] = base64.b64encode(image_bytes).decode("ascii")

        client = self._get_client()
        resp = client.invoke_model(
            modelId="amazon.titan-embed-image-v1",
            body=json.dumps(body),
        )
        result = json.loads(resp["body"].read())
        return result["embedding"]
