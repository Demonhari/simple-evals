
import random
import re

import pandas

from . import common
from .common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from datasets import load_dataset


subsets = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering', 'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']

class KMMLUEval(Eval):
    def __init__(self, num_examples: int | None = None):
        examples = []
        for subset in subsets:
            data = load_dataset(
                "HAERAE-HUB/KMMLU", subset, split="test", token=True
            )
            examples.extend([
                {
                    "Question": item["question"],
                    "A": item["A"],
                    "B": item["B"],
                    "C": item["C"],
                    "D": item["D"],
                    "Answer": chr(ord("A") + item["answer"] - 1),
                    "Subject": item["Category"],
                }
                for item in data
            ])
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        print(len(examples), "examples loaded from KMMLU")
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages))
            extracted_answer = None
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                match = re.search(regex, response_text)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, convo=convo
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
