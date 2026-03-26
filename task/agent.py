import json
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client = AsyncDial(base_url=self.endpoint, api_key="dial_api_key", api_version='2025-01-01-preview')
        coordination_stage = StageProcessor.open_stage(choice, "Coordination Request")
        coordination_request = await self.__prepare_coordination_request(client, request)
        coordination_stage.append_content(json.dumps(coordination_request.model_dump(), indent=2))
        StageProcessor.close_stage_safely(coordination_stage)

        agent_stage = StageProcessor.open_stage(choice, f"Agent Processing ({coordination_request.agent_name})")
        agent_message = await self.__handle_coordination_request(coordination_request, choice, agent_stage, request)
        StageProcessor.close_stage_safely(agent_stage)

        return await self.__final_response(client, choice, request, agent_message)

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            model=self.deployment_name,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return CoordinationRequest.model_validate(result)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            if msg.role == Role.USER and msg.custom_content:
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append(msg.dict(exclude_none=True))
        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        if coordination_request.agent_name == AgentName.UMS:
            gateway = UMSAgentGateway(self.ums_agent_endpoint)
            return await gateway.response(choice, stage, request, coordination_request.additional_instructions)
        else:
            gateway = GPAGateway(self.endpoint)
            return await gateway.response(choice, stage, request, coordination_request.additional_instructions)

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        last_content = messages[-1].get("content", "")
        augmented = f"Context from agent:\n{agent_message.content}\n\nUser request:\n{last_content}"
        messages[-1]["content"] = augmented

        response = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            model=self.deployment_name,
            stream=True
        )

        final_content = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                final_content += content
                choice.append_content(content)

        return Message(role=Role.ASSISTANT, content=final_content)
