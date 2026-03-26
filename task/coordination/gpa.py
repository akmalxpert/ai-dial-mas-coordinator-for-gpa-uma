from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        client = AsyncDial(base_url=self.endpoint, api_key="dial_api_key", api_version='2025-01-01-preview')

        messages = self.__prepare_gpa_messages(request, additional_instructions)
        response = await client.chat.completions.create(
            messages=messages,
            deployment_name="general-purpose-agent",
            model="general-purpose-agent",
            stream=True,
            extra_headers={'x-conversation-id': request.headers.get('x-conversation-id', '')}
        )

        content = ""
        result_custom_content = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            print(f"Delta: {delta}")

            if delta.content:
                content += delta.content
                stage.append_content(delta.content)

            if delta.custom_content:
                if delta.custom_content.attachments:
                    result_custom_content.attachments.extend(delta.custom_content.attachments)

                if delta.custom_content.state:
                    result_custom_content.state = delta.custom_content.state

                cc_dict = delta.custom_content.dict(exclude_none=True)
                if 'stages' in cc_dict:
                    for stg in cc_dict['stages']:
                        idx = stg['index']
                        if idx in stages_map:
                            if 'content' in stg:
                                stages_map[idx].append_content(stg['content'])
                            if 'attachments' in stg:
                                for att in stg['attachments']:
                                    stages_map[idx].add_attachment(Attachment(**att))
                            if stg.get('status') == 'completed':
                                StageProcessor.close_stage_safely(stages_map[idx])
                        else:
                            name = stg.get('name', f"GPA Stage {idx}")
                            new_stage = StageProcessor.open_stage(choice, name)
                            stages_map[idx] = new_stage
                            if 'content' in stg:
                                new_stage.append_content(stg['content'])

        if result_custom_content.attachments:
            for attachment in result_custom_content.attachments:
                choice.add_attachment(
                    Attachment(**attachment.dict(exclude_none=True))
                )

        choice.set_state({_IS_GPA: True, _GPA_MESSAGES: result_custom_content.state})

        return Message(role=Role.ASSISTANT, content=content)

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages = []

        for i in range(len(request.messages)):
            msg = request.messages[i]
            if msg.role == Role.ASSISTANT:
                if msg.custom_content and msg.custom_content.state:
                    state = msg.custom_content.state
                    if state.get(_IS_GPA) is True:
                        res_messages.append(request.messages[i - 1].dict(exclude_none=True))

                        copied = deepcopy(msg)
                        copied.custom_content.state = state.get(_GPA_MESSAGES)
                        res_messages.append(copied.dict(exclude_none=True))

        last_msg = request.messages[-1]
        res_messages.append(last_msg.dict(exclude_none=True))

        if additional_instructions:
            last_content = res_messages[-1].get("content", "")
            res_messages[-1]["content"] = f"{last_content}\n\nAdditional instructions: {additional_instructions}"

        return res_messages
