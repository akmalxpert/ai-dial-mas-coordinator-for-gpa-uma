import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice

_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        conversation_id = self.__get_ums_conversation_id(request)

        if conversation_id is None:
            conversation_id = await self.__create_ums_conversation()
            choice.set_state({_UMS_CONVERSATION_ID: conversation_id})

        last_message = request.messages[-1]
        user_content = last_message.content
        if additional_instructions:
            user_content = f"{user_content}\n\nAdditional instructions: {additional_instructions}"

        response_content = await self.__call_ums_agent(conversation_id, user_content, stage)

        return Message(role=Role.ASSISTANT, content=response_content)

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for msg in request.messages:
            if msg.custom_content and msg.custom_content.state:
                state = msg.custom_content.state
                if _UMS_CONVERSATION_ID in state:
                    return state[_UMS_CONVERSATION_ID]
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations",
                json={"title": "UMS Agent Conversation"},
                timeout=30.0
            )
            data = response.json()
            return data["id"]

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        accumulated = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                    "POST",
                    f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                    json={"message": {"role": "user", "content": user_message}, "stream": True}
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")
                            if content:
                                accumulated += content
                                stage.append_content(content)
                    except json.JSONDecodeError:
                        continue
        return accumulated
