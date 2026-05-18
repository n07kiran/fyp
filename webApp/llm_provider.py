import asyncio
import httpx
import os
import base64
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Protocol, Optional

load_dotenv()


@dataclass(frozen=True)
class ApiKeySelection:
    value: str
    index: int
    total: int


class RoundRobinKeyRotator:
    def __init__(self, api_keys: list[str] | tuple[str, ...]):
        self._api_keys = [key.strip() for key in api_keys if key.strip()]
        self._cursor = 0
        self._lock = asyncio.Lock()

    @property
    def has_keys(self) -> bool:
        return bool(self._api_keys)

    @property
    def total(self) -> int:
        return len(self._api_keys)

    async def next_key(self) -> ApiKeySelection:
        if not self._api_keys:
            raise RuntimeError("No Gemini API keys configured.")

        async with self._lock:
            index = self._cursor
            self._cursor = (self._cursor + 1) % len(self._api_keys)

        return ApiKeySelection(
            value=self._api_keys[index],
            index=index,
            total=len(self._api_keys),
        )


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    fallback_text: str
    image_b64: Optional[str] = None
    gradcam_b64: Optional[str] = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str


class LLMProvider(Protocol):
    name: str
    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...


class LLMRateLimitError(RuntimeError):
    def __init__(
        self,
        *,
        provider: str,
        attempted_keys: int,
        total_keys: int,
        retry_after_seconds: float | None = None,
        last_error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.attempted_keys = attempted_keys
        self.total_keys = total_keys
        self.retry_after_seconds = retry_after_seconds
        self.last_error = last_error

        message = f"{provider} rate limited after trying {attempted_keys}/{total_keys} API keys."
        if retry_after_seconds is not None:
            message += f" Retry after ~{int(retry_after_seconds)}s."
        if last_error is not None:
            message += f" ({last_error})"

        super().__init__(message)


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_keys: list[str] | tuple[str, ...], model: str):
        self._rotator = RoundRobinKeyRotator(api_keys)
        self._model = model
        self._timeout = httpx.Timeout(30.0, connect=10.0)

    @property
    def has_keys(self) -> bool:
        return self._rotator.has_keys

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._rotator.has_keys:
            return LLMResponse(text=request.fallback_text, provider="fallback_no_keys")

        last_error: Exception | None = None
        attempts = max(1, self._rotator.total)
        rate_limited_attempts = 0
        retry_after_seconds: float | None = None

        for _ in range(attempts):
            selected_key = await self._rotator.next_key()
            try:
                text = await self._call_gemini(selected_key.value, request)
                if text:
                    return LLMResponse(text=text, provider=f"{self.name}:{self._model}")
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code == 429:
                    rate_limited_attempts += 1
                    retry_after_seconds = max(
                        retry_after_seconds or 0.0,
                        _parse_retry_after_seconds(exc.response.headers.get("Retry-After")) or 0.0,
                    )

                if status_code not in {401, 403, 429, 500, 502, 503, 504}:
                    break
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc

        print(f"Gemini API failed. Falling back. Error: {last_error}")
        return LLMResponse(text=request.fallback_text, provider="fallback_error")

    async def _call_gemini(self, api_key: str, request: LLMRequest) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        
        parts = [{"text": request.user_prompt}]
        
        if request.image_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": request.image_b64
                }
            })
            
        if request.gradcam_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": request.gradcam_b64
                }
            })

        payload = {
            "systemInstruction": {
                "parts": [{"text": request.system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
            },
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts).strip()


def build_provider() -> LLMProvider:
    api_keys_str = os.getenv("GEMINI_API_KEYS", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    if api_keys:
        return GeminiProvider(api_keys, model)
        
    class MockProvider:
        name = "mock"
        async def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(text=request.fallback_text, provider="mock")
            
    return MockProvider()
