import json
import logging
from typing import TYPE_CHECKING
from typing import Literal

import gitlab
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .embeddings import NebiusEmbeddings
from .settings import ClaireSettings
from .settings import GitlabSettings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.vectorstores import VectorStoreRetriever

LOG = logging.getLogger(__name__)


class MRThread(BaseModel):
    discussion_type: Literal["suggestion", "question", "issue", "nitpick", "note", "chore", "thought"] | None
    blocking: bool
    author: str
    resolved: bool
    message: str
    related_file: str | None
    related_line: int | None
    related_diff: str | None
    replies: list[str] = []


class GitlabAdapter:
    def __init__(self, settings: GitlabSettings) -> None:
        self._settings = settings
        self._gitlab = gitlab.Gitlab(
            url=str(self._settings.url),
            private_token=self._settings.token.get_secret_value(),
            ssl_verify=False,
        )
        self._project = self._gitlab.projects.get(self._settings.project_id)

    def list_merge_request_threads(self, merge_request_id: int, ignore_authors: set[str]) -> list[MRThread]:  # noqa: C901
        """Get all discussion threads from a merge request.

        Args:
            merge_request_id: The ID of the merge request
            ignore_authors: The authors to ignore

        Returns:
            List of MRThread objects representing discussion threads
        """
        merge_request = self._project.mergerequests.get(merge_request_id)

        threads = []

        for discussion in merge_request.discussions.list(iterator=True):
            if not discussion.notes:
                continue

            # Get the first note (thread starter)
            discussion_notes = discussion.attributes["notes"]
            first_note = discussion_notes[0]

            # Skip system notes
            if first_note["system"] or first_note["internal"]:
                continue

            # Get author and check if discussion should be skipped
            author = first_note["author"]["username"]
            if author in ignore_authors:
                continue

            # Determine if blocking based on message content
            message_lower = first_note["body"].lower()
            if "non-blocking" in message_lower:
                blocking = False
            elif "blocking" in message_lower:
                blocking = True
            else:
                blocking = True  # Default to blocking

            # Determine comment type based on message content
            discussion_type = self._classify_discussion_type(message_lower)

            # Get related change if the note is attached to a specific line
            related_file = None
            related_diff = None
            related_line = None
            if "position" in first_note:
                position = first_note["position"]
                related_file = position.get("new_path")
                related_line = int(position.get("new_line") or "0")

                if related_file and related_line:
                    # Get the diff content that this note relates to
                    changes = merge_request.changes()
                    for change in changes["changes"]:
                        if change["new_path"] == related_file:
                            # Extract the relevant diff section around the line
                            related_diff = change["diff"]
                            break

            # Collect replies (subsequent notes in the discussion)
            replies = []
            if len(discussion_notes) > 1:
                replies = [note["body"] for note in discussion_notes[1:]]

            thread = MRThread(
                discussion_type=discussion_type,
                blocking=blocking,
                author=author,
                resolved=first_note.get("resolved", True),
                message=first_note["body"],
                related_file=related_file,
                related_line=related_line,
                related_diff=related_diff,
                replies=replies,
            )

            threads.append(thread)

        return threads

    def _classify_discussion_type(self, message: str) -> str | None:
        """Classify discussion type based on message content."""
        # TODO(tretiak-rd): Use AI for that
        message_lower = message.lower()

        discussion_patterns = [
            (["suggestion"], "suggestion"),
            (["question"], "question"),
            (["issue"], "issue"),
            (["nitpick"], "nitpick"),
            (["chore"], "chore"),
            (["todo"], "todo"),
            (["thought"], "thought"),
        ]

        for patterns, discussion_type in discussion_patterns:
            if any(word in message_lower for word in patterns):
                return discussion_type

        return None

    def list_merged_merge_requests(self, authors: set[str]) -> list[int]:
        """Get a list of merged Merge Request IDs by specified authors.

        Args:
            authors: List of author usernames to filter by

        Returns:
            List of merge request IDs that were merged by the specified authors
        """
        merge_request_ids = []

        for author in authors:
            mrs = self._project.mergerequests.list(author_username=author, state="merged", all=True)
            merge_request_ids.extend(mr.iid for mr in mrs)

        return merge_request_ids

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
        self._embeddings = NebiusEmbeddings(api_key=self._settings.openai.api_key)
        self._memory: VectorStore = Chroma(persist_directory=".chroma_store", embedding_function=self._embeddings)
        self._retriever: VectorStoreRetriever = self._memory.as_retriever()

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

    def index_merge_request_discussions(self, merge_request_id: int, ignore_authors: set[str]) -> None:
        threads = self._gitlab.list_merge_request_threads(merge_request_id, ignore_authors)
        prompt_template = """When we change {related_file} on line {related_line} with diff
        ```
        {related_diff}
        ```
        there was the following {blocking} {discussion_type}:
        {message}
        {discussion}
        """
        documents = [
            Document(
                page_content=prompt_template.format(
                    related_file=thread.related_file,
                    related_line=thread.related_line,
                    related_diff=thread.related_diff,
                    blocking=thread.blocking,
                    discussion_type=thread.discussion_type,
                    message=thread.message,
                    discussion="\n".join(thread.replies),
                ),
                metadata={"file": thread.related_file, "blocking": thread.blocking},
            )
            for thread in threads
        ]
        self._memory.add_documents(documents)

    def ask(self, question: str) -> str:
        system_prompt = SystemMessage(
            "You are a project wisdom keeper, your memory is vector storage. You need to answer the question retrieving"
            "information from it using queries."
            "Generate a search query for the vector storage to answer the question"
            ", you will get documents by your query."
            "Respond with either QUERY response if you need to search something or ANSWER if you are ready to answer. "
            "No more than 400 characters in ANSWER. "
            "No more than 80 characters in QUERY. "
            "Respond with JSON in this exact format: "
            "in case of QUERY:"
            '{"query": "Brief query for the vector storage"}'
            "or in case of ANSWER:"
            '{"answer": "Comprehensive answer"}'
        )
        human_prompt = HumanMessage(f"Question: {question}")

        all_documents: list[Document] = []
        for _ in range(3):
            response = self._model.invoke([system_prompt, human_prompt]).content
            response_data = json.loads(response.removeprefix("```json\n").removesuffix("```"))
            if "query" in response_data:
                docs = self._retriever.invoke(response_data["query"])
                all_documents.extend(docs)
                context = "\n\n".join([doc.page_content for doc in docs])
                human_prompt = HumanMessage(f"Question: {question}\n\nDocuments: {context}")
            elif "answer" in response_data:
                return response_data["answer"]

        response = self._model.invoke([system_prompt, human_prompt, HumanMessage("You MUST answer now!")]).content
        response_data = json.loads(response.removeprefix("```json\n").removesuffix("```"))
        return response_data["answer"]

    def _init_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.openai.model,
            api_key=self._settings.openai.api_key,
            base_url=str(self._settings.openai.base_url),
            temperature=self._settings.openai.temperature,
        )
