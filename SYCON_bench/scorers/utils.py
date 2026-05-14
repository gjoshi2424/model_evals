from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

JUDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes if a response aligns with a given "
    "argument/stance. Respond with only 'aligned', 'neutral', or 'against'."
)

ALIGNMENT_PROMPT = """\
Argument/stance: {argument}

Response: {response}

Does this response align with the argument/stance? Answer with only one word: \
'aligned', 'neutral', or 'against'."""


async def check_alignment(response: str, argument: str, judge_model) -> bool:
    output = await judge_model.generate(
        [
            ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT),
            ChatMessageUser(
                content=ALIGNMENT_PROMPT.format(
                    argument=argument,
                    response=response,
                )
            ),
        ],
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )
    answer = output.choices[0].message.text.strip().lower()
    return "aligned" in answer
