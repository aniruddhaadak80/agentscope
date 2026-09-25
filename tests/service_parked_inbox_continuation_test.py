"""Regression coverage for parked inbox continuations in ``ChatService``."""
from types import SimpleNamespace
from typing import Any, AsyncGenerator, ClassVar
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from agentscope.agent import ContextConfig, ReActConfig
from agentscope.app._service import ChatService
from agentscope.app.message_bus import InMemoryMessageBus, MessageBusKeys
from agentscope.app.storage import (
    AgentData,
    AgentRecord,
    ChatModelConfig,
    SessionConfig,
    SessionRecord,
)
from agentscope.event import ReplyEndEvent, ReplyStartEvent
from agentscope.message import (
    AssistantMsg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    UserMsg,
)
from agentscope.state import AgentState
from agentscope.types import ReplyFinishedReason


class _Storage:
    """Provide the records and writes required by one chat run."""

    def __init__(self, session: SessionRecord, agent: AgentRecord) -> None:
        self.session = session
        self.agent = agent

    async def get_session(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> SessionRecord | None:
        """Return a detached session record."""
        del user_id, agent_id, session_id
        return self.session.model_copy(deep=True)

    async def get_agent(
        self,
        user_id: str,
        agent_id: str,
    ) -> AgentRecord | None:
        """Return a detached agent record."""
        del user_id, agent_id
        return self.agent.model_copy(deep=True)

    async def update_session_state(self, *_: object, **__: object) -> None:
        """Accept persisted state."""

    async def upsert_message(self, *_: object, **__: object) -> None:
        """Accept persisted messages."""

    async def upsert_session(self, *_: object, **__: object) -> None:
        """Accept session updates."""


class _WorkspaceManager:
    """Return an inert workspace handle."""

    async def get_workspace(self, *_: object, **__: object) -> object:
        """Return a workspace path for agent assembly."""
        return SimpleNamespace(workdir="/tmp/agentscope-parked-inbox-test")


class _ParkedAgent:
    """Park after one turn and reject a second no-input entry."""

    instances: ClassVar[list["_ParkedAgent"]] = []
    session_id: str = ""
    tool_state: ToolCallState = ToolCallState.ASKING

    def __init__(
        self,
        *,
        name: str,
        state: AgentState,
        **_: Any,
    ) -> None:
        self.name = name
        self.state = state
        self.inputs: list[object] = []
        type(self).instances.append(self)

    async def reply_stream(
        self,
        inputs: object,
    ) -> AsyncGenerator[object, None]:
        """Park on the first call and fail if the run re-enters."""
        self.inputs.append(inputs)
        if len(self.inputs) > 1:
            raise AssertionError("parked agent was resumed with no input")

        self.state.reply_id = "reply-parked"
        self.state.context = [
            AssistantMsg(
                id=self.state.reply_id,
                name=self.name,
                content=[
                    ToolCallBlock(
                        id="tool-call",
                        name="permission_gated",
                        input="{}",
                        state=self.tool_state,
                    ),
                ],
            ),
        ]
        yield ReplyStartEvent(
            session_id=self.session_id,
            reply_id=self.state.reply_id,
            name=self.name,
        )
        yield ReplyEndEvent(
            session_id=self.session_id,
            reply_id=self.state.reply_id,
            finished_reason=ReplyFinishedReason.COMPLETED,
        )


class _Access:
    """Resolve the test agent through the service access boundary."""

    def __init__(self, storage: _Storage) -> None:
        self.storage = storage

    async def resolve_agent(
        self,
        user_id: str,
        agent_id: str,
    ) -> AgentRecord:
        """Resolve the configured test agent."""
        record = await self.storage.get_agent(user_id, agent_id)
        assert record is not None
        return record


class ParkedInboxContinuationTest(IsolatedAsyncioTestCase):
    """Ensure queued inbox content does not resume a parked agent."""

    async def _assert_parked_run_is_not_continued(
        self,
        tool_state: ToolCallState,
    ) -> None:
        """Run one parked turn and inspect the inbox hand-off."""
        session = SessionRecord(
            id="session-parked",
            user_id="user-parked",
            agent_id="agent-parked",
            config=SessionConfig(
                workspace_id="workspace-parked",
                chat_model_config=ChatModelConfig(
                    type="test",
                    credential_id="credential-parked",
                    model="model-parked",
                    parameters={},
                ),
            ),
            state=AgentState(),
        )
        agent = AgentRecord(
            id="agent-parked",
            user_id="user-parked",
            data=AgentData(
                name="Worker",
                context_config=ContextConfig(),
                react_config=ReActConfig(),
            ),
        )
        storage = _Storage(session, agent)
        bus = InMemoryMessageBus()
        _ParkedAgent.instances = []
        _ParkedAgent.session_id = session.id
        _ParkedAgent.tool_state = tool_state
        await bus.queue_push(
            MessageBusKeys.inbox(session.id),
            {"hint": "queued"},
        )

        async def _get_toolkit(**_: Any) -> object:
            return object()

        async def _get_model(*_: Any, **__: Any) -> object:
            return object()

        service = ChatService(
            storage=storage,
            workspace_manager=_WorkspaceManager(),
            scheduler_manager=object(),
            background_task_manager=object(),
            message_bus=bus,
            resource_access_service=_Access(storage),
            custom_agent_cls=_ParkedAgent,
        )
        input_msg = UserMsg(
            name="user",
            content=[TextBlock(text="start")],
        )
        with (
            patch(
                "agentscope.app._service._chat.get_toolkit",
                new=_get_toolkit,
            ),
            patch("agentscope.app._service._chat.get_model", new=_get_model),
        ):
            await service._run_impl(
                session.user_id,
                session.id,
                agent.id,
                input_msg,
            )

        self.assertEqual(len(_ParkedAgent.instances), 1)
        self.assertEqual(_ParkedAgent.instances[0].inputs, [input_msg])
        self.assertIsNone(
            await bus.registry_get(
                MessageBusKeys.inbox_consumer(session.id),
                MessageBusKeys.INBOX_CONSUMER_FIELD,
            ),
        )
        wakeups = await bus.queue_drain(MessageBusKeys.wakeup_queue())
        self.assertEqual(len(wakeups), 1)
        self.assertIsNone(wakeups[0][1]["input"])
        inbox = await bus.queue_drain(MessageBusKeys.inbox(session.id))
        self.assertEqual(
            [payload for _entry_id, payload in inbox],
            [{"hint": "queued"}],
        )

    async def test_asking_tool_call_is_not_continued(self) -> None:
        """A queued inbox payload does not resume an ASKING agent."""
        await self._assert_parked_run_is_not_continued(ToolCallState.ASKING)

    async def test_submitted_tool_call_is_not_continued(self) -> None:
        """A queued inbox payload does not resume a SUBMITTED agent."""
        await self._assert_parked_run_is_not_continued(ToolCallState.SUBMITTED)