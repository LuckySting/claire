from typing import Annotated

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class OpenAISettings(BaseModel):
    base_url: HttpUrl
    api_key: SecretStr
    model: str
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7


class GitlabSettings(BaseModel):
    url: HttpUrl
    token: SecretStr
    project_id: int


class ClaireSettings(BaseSettings):
    openai: OpenAISettings
    gitlab: GitlabSettings
