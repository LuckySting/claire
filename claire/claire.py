import json
import logging

import gitlab
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .settings import ClaireSettings
from .settings import GitlabSettings

LOG = logging.getLogger(__name__)


class GitlabAdapter:
    def __init__(self, settings: GitlabSettings) -> None:
        self._settings = settings
        self._gitlab = gitlab.Gitlab(
            url=str(self._settings.url),
            private_token=self._settings.token.get_secret_value(),
            ssl_verify=False,
        )
        self._project = self._gitlab.projects.get(self._settings.project_id)

    def get_merge_request_diff(self, merge_request_id: int) -> str:
        """Get the diff of a merge request.

        Args:
            merge_request_id: The ID of the merge request

        Returns:
            The diff as a string
        """
        mr = self._project.mergerequests.get(merge_request_id)
        changes = mr.changes()

        return "\n".join(change["diff"] for change in changes["changes"])

    def add_merge_request_comment(self, merge_request_id: int, comment: str) -> None:
        """Add a comment to a merge request.

        Args:
            merge_request_id: The ID of the merge request
            comment: The comment text to add
        """
        mr = self._project.mergerequests.get(merge_request_id)
        mr.notes.create({"body": comment})


class MRSummary(BaseModel):
    title: str
    summary: str


class Claire:
    def __init__(self, settings: ClaireSettings) -> None:
        self._settings = settings
        self._gitlab = GitlabAdapter(self._settings.gitlab)
        self._model = self._init_model()

    def get_merge_request_summary(self, merge_request_id: int) -> MRSummary:
        """Generate an AI-powered summary of a merge request.

        Args:
            merge_request_id: The ID of the merge request to summarize

        Returns:
            MRSummary object containing the title and summary of the changes

        Raises:
            json.JSONDecodeError: If the AI response cannot be parsed as JSON
        """
        prompt = SystemMessage(
            "You are a code reviewer analyzing a merge request diff. "
            "Analyze the provided diff and create a concise title and summary. "
            "Use Markdown as format. Use lists if needed. "
            "No more than 400 characters in summary and 80 characters in title. "
            "Respond with JSON in this exact format: "
            '{"title": "Brief descriptive title", "summary": "Detailed summary of changes"}'
        )
        mr_diff = self._gitlab.get_merge_request_diff(merge_request_id)
        diff_message = HumanMessage(mr_diff)

        result = self._model.invoke([prompt, diff_message])
        result_json_str = result.content.removeprefix("```json\n").removesuffix("```")

        try:
            response_data = json.loads(result_json_str)
        except json.decoder.JSONDecodeError:
            LOG.exception("Could not decode JSON response: %s", result_json_str)
            raise
        return MRSummary(title=response_data["title"], summary=response_data["summary"])

    def summarize_merge_request(self, merge_request_id: int) -> None:
        mr_summary = self.get_merge_request_summary(merge_request_id)
        comment = [
            "Claire is here! I summarized this MergeRequest for you 😉",
            f"# {mr_summary.title}",
            mr_summary.summary,
        ]
        self._gitlab.add_merge_request_comment(merge_request_id, "\n".join(comment))

    def _init_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.openai.model,
            api_key=self._settings.openai.api_key,
            base_url=str(self._settings.openai.base_url),
            temperature=self._settings.openai.temperature,
        )
