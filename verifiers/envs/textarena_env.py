import random
import re
from copy import deepcopy
from typing import Any, Callable
import os

try:
    import nltk  # type: ignore
except ImportError:
    print("nltk is not installed. Please install it with `uv pip install nltk`.")
    exit(1)

from datasets import Dataset

# monkey-patch nltk.download to always be quiet before importing textarena
_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(  # type: ignore[invalid-assignment]
    *args, **{**kwargs, "quiet": True}
)

try:
    import textarena as ta  # type: ignore
except ImportError:
    print(
        "textarena is not installed. Please install it with `uv pip install textarena`."
    )
    exit(1)

from verifiers.envs.multiturn_env import MultiTurnEnv  # noqa: E402
from verifiers.parsers.xml_parser import XMLParser  # noqa: E402
from verifiers.rubrics.rubric import Rubric  # noqa: E402
from verifiers.types import (  # noqa: E402
    Messages,
    State,
)


def clean_assistant_message(content: str) -> str:
    """
    Remove reasoning tokens (<think>...</think> or <tools>...</tools>) from assistant message
    to reduce context length during rollout.
    Preserves solution markers.
    Handles malformed XML where LLM outputs <thinkContent or <toolsContent without closing >.
    Also handles alternative closing tag <|end_internal_thought|>.
    """
    # Remove both <think> and <tools> tags and their content
    # Handles </think>, </tools>, or <|end_internal_thought|> as closing tags
    content = re.sub(r'<(?:think|tools).*?(?:</(?:think|tools)>|<\|end_internal_thought\|>)', '', content, flags=re.DOTALL)
    # Clean up excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content.strip()


class TextArenaEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """

    def __init__(
        self,
        game: str = "Wordle-v0",
        num_train_examples: int = 1000,
        num_eval_examples: int = 0,
        system_prompt: str | None = None,
        parser: XMLParser | None = None,
        rubric: Rubric | None = None,
        feedback_fn: Callable[[str], str] = lambda x: x,
        seed: int = 0,
        **kwargs,
    ):
        # default parser in textarena is XMLParser
        parser = parser or XMLParser(fields=["think", "guess"], answer_field="guess")

        self.game = game
        self.ta_env = ta.make(env_id=game)
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        self.feedback_fn = feedback_fn

        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        dataset, eval_dataset = self.ta_to_hf()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )

    async def is_completed(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> bool:
        completed = await super().is_completed(messages, state, **kwargs)
        if "is_completed" in state and state["is_completed"]:
            state.pop("ta_env")
            return state["is_completed"]
        return False or completed

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> tuple[Messages, State]:
        # Clean the last assistant message to reduce context length
        if os.environ.get("is_train") and messages and isinstance(messages[-1], dict):
            if messages[-1].get("role") == "assistant" and "content" in messages[-1]:
                messages[-1]["content"] = clean_assistant_message(
                    messages[-1]["content"]
                )
        
        # load env
        if "ta_env" not in state:
            ta_env = deepcopy(self.ta_env)
            ta_env.reset(num_players=1)
            if os.environ.get("is_train"):
                # FIX: Set secret word at the WordleEnv level (not wrapper level)
                # TextArena uses lowercase for secret words
                ta_env.env.env.state.game_state["secret_word"] = state["answer"].lower()
            state["ta_env"] = ta_env
        else:
            ta_env = state["ta_env"]
        # parse guess
        assert isinstance(messages[-1], dict)
        guess = self.parser.parse_answer(messages)
        # step env
        is_completed, _ = ta_env.step(str(guess))
        state["is_completed"] = is_completed
        _, observation = ta_env.get_observation()
        
        if os.environ.get("is_train"):
            # Handle case where game ends without feedback in observation
            # When is_completed=True, TextArena doesn't include feedback in observation
            # but it IS correctly stored in guess_history
            # Check if the observation has feedback for the most recent guess
            guess_history = ta_env.state.game_state.get("guess_history", [])
            if is_completed and guess_history:
                last_guess, feedback_list = guess_history[-1]
                expected_feedback_line = " ".join(last_guess.upper())
                
                # If the last guess feedback is not in the observation, add it
                if expected_feedback_line not in observation:
                    feedback_text = f"\n[GAME] You submitted [{last_guess}].\nFeedback:\n"
                    feedback_text += expected_feedback_line + "\n"
                    feedback_text += " ".join(feedback_list)
                    observation += feedback_text
        
        feedback = self.feedback_fn(observation)
        
        if os.environ.get("is_train"):
            # FIX: Increment turn counter to match MultiTurnEnv behavior
            # This ensures proper turn counting for environments that rely on it
            state["turn"] = state.get("turn", 0) + 1
        
        return [{"role": "user", "content": str(feedback)}], state

    def ta_to_hf(self) -> tuple[Dataset, Dataset | None]:
        dataset_rows = []
        eval_dataset_rows = []
        ta_env = ta.make(env_id=self.game)
        ta_env.reset(num_players=1)
        _, user_prompt = ta_env.get_observation()
        words = ta_env.word_list
        # set seed
        random.seed(self.seed)
        for i in range(self.num_train_examples + self.num_eval_examples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_train_examples:
                dataset_rows.append({"question": question, "answer": answer})
            else:
                eval_dataset_rows.append({"question": question, "answer": answer})
        dataset = Dataset.from_list(dataset_rows)
        if self.num_eval_examples > 0:
            eval_dataset = Dataset.from_list(eval_dataset_rows)
        else:
            eval_dataset = None
        return dataset, eval_dataset
