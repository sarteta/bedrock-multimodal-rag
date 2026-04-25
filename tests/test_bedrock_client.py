from unittest.mock import MagicMock, patch

import pytest

from bedrock_rag.bedrock_client import BedrockClient


def test_client_construction_does_not_call_aws():
    """We should be able to construct without AWS creds -- boto is lazy."""
    c = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    assert c.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert c._client is None


def test_converse_calls_boto_with_proper_shape():
    c = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    fake = MagicMock()
    fake.converse.return_value = {
        "output": {"message": {"content": [{"text": "hello"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 3},
    }
    c._client = fake

    out = c.converse(
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        system="you are a helpful assistant",
        max_tokens=500,
    )

    assert out.text == "hello"
    assert out.input_tokens == 10
    assert out.output_tokens == 3
    fake.converse.assert_called_once()
    args = fake.converse.call_args.kwargs
    assert args["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert args["inferenceConfig"]["maxTokens"] == 500
    assert args["system"][0]["text"] == "you are a helpful assistant"


def test_converse_concatenates_multi_chunk_response():
    c = BedrockClient(model_id="m")
    fake = MagicMock()
    fake.converse.return_value = {
        "output": {"message": {"content": [{"text": "chunk1 "}, {"text": "chunk2"}]}},
        "usage": {"inputTokens": 5, "outputTokens": 2},
    }
    c._client = fake

    out = c.converse(messages=[])
    assert out.text == "chunk1 chunk2"


def test_converse_omits_system_when_none():
    c = BedrockClient(model_id="m")
    fake = MagicMock()
    fake.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {},
    }
    c._client = fake

    c.converse(messages=[])
    args = fake.converse.call_args.kwargs
    assert "system" not in args


def test_embed_text_calls_titan_text_v2():
    c = BedrockClient(model_id="m")
    fake = MagicMock()
    body_stream = MagicMock()
    body_stream.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    fake.invoke_model.return_value = {"body": body_stream}
    c._client = fake

    vec = c.embed_text("hello")
    assert vec == [0.1, 0.2, 0.3]
    args = fake.invoke_model.call_args.kwargs
    assert args["modelId"] == "amazon.titan-embed-text-v2:0"


def test_embed_multimodal_requires_text_or_image():
    c = BedrockClient(model_id="m")
    with pytest.raises(ValueError, match="must provide text or image"):
        c.embed_multimodal()


def test_embed_multimodal_with_image_only():
    c = BedrockClient(model_id="m")
    fake = MagicMock()
    body_stream = MagicMock()
    body_stream.read.return_value = b'{"embedding": [0.5] * 1024}'
    body_stream.read.return_value = b'{"embedding": [0.5]}'
    fake.invoke_model.return_value = {"body": body_stream}
    c._client = fake

    vec = c.embed_multimodal(image_bytes=b"\x00\x01\x02")
    assert vec == [0.5]
    args = fake.invoke_model.call_args.kwargs
    assert args["modelId"] == "amazon.titan-embed-image-v1"


def test_embed_multimodal_with_text_and_image():
    c = BedrockClient(model_id="m")
    fake = MagicMock()
    body_stream = MagicMock()
    body_stream.read.return_value = b'{"embedding": [0.1]}'
    fake.invoke_model.return_value = {"body": body_stream}
    c._client = fake

    import json
    c.embed_multimodal(text="caption", image_bytes=b"img")
    args = fake.invoke_model.call_args.kwargs
    body = json.loads(args["body"])
    assert "inputText" in body
    assert "inputImage" in body
