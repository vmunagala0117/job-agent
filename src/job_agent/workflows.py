from __future__ import annotations

from uuid import uuid4

from agent_framework import (
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from typing_extensions import Never


class CoordinatorExecutor(Executor):
    """Single-node coordinator that delegates reasoning to a chat agent."""

    def __init__(self, client: AzureOpenAIChatClient, id: str = "coordinator"):
        self.agent = client.create_agent(
            name="JobCoordinator",
            instructions=(
                "You are a job search assistant. Keep answers concise. "
                "For now, echo back requested actions and outline next steps. "
                "Avoid making up recruiter contacts or job URLs."
            ),
        )
        super().__init__(id=id)

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Never, str]) -> None:
        # Run the agent and forward assistant messages as streaming updates
        response = await self.agent.run(messages)
        for message in response.messages:
            if message.role == Role.ASSISTANT and message.contents:
                await ctx.add_event(
                    AgentRunUpdateEvent(
                        self.id,
                        data=AgentRunResponseUpdate(
                            contents=[TextContent(text=message.contents[-1].text)],
                            role=Role.ASSISTANT,
                            response_id=str(uuid4()),
                        ),
                    )
                )
        # Yield a simple string output for the HTTP response body
        await ctx.yield_output(response.text)


def build_agent(client: AzureOpenAIChatClient):
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: CoordinatorExecutor(client), name="coordinator")
        .set_start_executor("coordinator")
        .build()
    )
    return workflow.as_agent()
