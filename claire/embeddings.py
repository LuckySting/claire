"""Wrapper around Nebius AI Studio's Embeddings API."""

import logging
import warnings
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import Self

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils import secret_from_env
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator

logger = logging.getLogger(__name__)


class NebiusEmbeddings(BaseModel, Embeddings):
    """Nebius embedding model integration.

    Setup:
        Install ``langchain_nebius`` and set environment variable
        ``NEBIUS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain_nebius
            export NEBIUS_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Nebius AI Studio model to use.

    Key init args — client params:
      api_key: Optional[SecretStr]

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from __module_name__ import NebiusEmbeddings

            embed = NebiusEmbeddings(
                model="BAAI/bge-en-icl",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "BAAI/bge-en-icl"
    """Embeddings model name to use.
    Instead, use 'BAAI/bge-en-icl' for example.
    """
    dimensions: int | None = None
    """The number of dimensions the resulting output embeddings should have.

    Not yet supported.
    """
    nebius_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("NEBIUS_API_KEY", default=None),
    )
    """Nebius AI API key.

    Automatically read from env variable `NEBIUS_API_KEY` if not provided.
    """
    nebius_api_base: str = Field(
        default_factory=from_env("NEBIUS_API_BASE", default="https://api.studio.nebius.ai/v1/"),
        alias="base_url",
    )
    """Endpoint URL to use."""
    embedding_ctx_length: int = 4096
    """The maximum number of tokens to embed at once.

    Not yet supported.
    """
    allowed_special: Literal["all"] | set[str] = set()
    """Not yet supported."""
    disallowed_special: Literal["all"] | set[str] | Sequence[str] = "all"
    """Not yet supported."""
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch.

    Not yet supported.
    """
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: float | tuple[float, float] | Any | None = Field(default=None, alias="timeout")
    """Timeout for requests to Nebius embedding API. Can be float, httpx.Timeout or
        None."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding.

    Not yet supported.
    """
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping.

    Not yet supported."""
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Any | None = None
    """Optional httpx.Client. Only used for sync invocations. Must specify
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Any | None = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify
        http_client as well if you'd like a custom client for sync invocations."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Logic that will post Pydantic initialization."""
        client_params: dict = {
            "api_key": (self.nebius_api_key.get_secret_value() if self.nebius_api_key else None),
            "base_url": self.nebius_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client} if self.http_client else {}
            self.client = openai.OpenAI(**client_params, **sync_specific).embeddings
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client} if self.http_async_client else {}
            self.async_client = openai.AsyncOpenAI(**client_params, **async_specific).embeddings
        return self

    @property
    def _invocation_params(self) -> dict[str, Any]:
        params: dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts using passage model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        for text in texts:
            response = self.client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = self.client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts using passage model asynchronously.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        for text in texts:
            response = await self.async_client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = await self.async_client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]
