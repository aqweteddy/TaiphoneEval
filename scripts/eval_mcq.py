import asyncio
import json
import random
from collections import Counter
from typing import Literal

from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

PROMPT1_EN = """
Think step by step and answer the following question.

{query}
""".strip()

PROMPT1_ZH = """
請逐步推理，回答以下問題。

{query}
""".strip()


PROMPT2 = """
告訴我這段落所說的答案選項是哪一個，以json格式回答。

{{
    "answer": "你的選項"
}}

## 段落

{reason}
""".strip()

random.seed(42)


class ModelResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"]


async def run_one(
    cli: AsyncOpenAI,
    query: str,
    model: str,
    sem,
    gpt_cli: AsyncOpenAI = None,
    lang: str = "en",
) -> tuple[list[dict[str, str]], str]:
    gpt_cli = gpt_cli or cli

    PROMPT1 = PROMPT1_EN if lang == "en" else PROMPT1_ZH

    async with sem:
        for _ in range(3):
            try:
                prompt = PROMPT1.format(query=query)
                resp = await cli.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2200,
                )
                resp_str = resp.choices[0].message.content

                prompt2 = PROMPT2.format(reason=resp_str).strip()
                messages = [
                    {"role": "user", "content": prompt2},
                ]
                model = "gpt-4.1-nano" if gpt_cli is not None else model
                resp = await gpt_cli.beta.chat.completions.parse(
                    model=model, messages=messages, response_format=ModelResponse
                )
                answer: ModelResponse = resp.choices[0].message.parsed
                messages.append({"role": "assistant", "content": f"{answer.answer}"})
                return [{"role": "assistant", "content": f"{resp_str}"}, messages[-1]]
            except Exception as e:
                print(e)
                continue
    return [{"content": "error"}, {"content": ""}]


def run_all(
    cli: AsyncOpenAI,
    questions: list[str],
    model: str,
    concurrency: int = 200,
    use_gpt_as_parser: bool = False,
    lang: str = "en",
):
    if use_gpt_as_parser:
        gpt_cli = AsyncOpenAI()
    else:
        gpt_cli = None

    async def arun_all():
        return await atqdm.gather(
            *[run_one(cli, q, model, sem, gpt_cli, lang) for q in questions],
        )

    sem = asyncio.Semaphore(concurrency)
    return asyncio.run(arun_all())


def get_model_name(url: str):
    cli = OpenAI(base_url=url)
    models = list(cli.models.list())
    return models[0].id


def format_query(x: dict[str, str]):
    result = """{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}""".format(**x)
    return {
        "_prompt": result,
    }


def rand_choices(x: dict[str, str]):
    abcs = [x["A"], x["B"], x["C"], x["D"]]
    answer = {
        "A": abcs[0],
        "B": abcs[1],
        "C": abcs[2],
        "D": abcs[3],
    }[x["answer"]]
    random.shuffle(abcs)
    return {
        "question": x["question"],
        "A": abcs[0],
        "B": abcs[1],
        "C": abcs[2],
        "D": abcs[3],
        "answer": "ABCD"[abcs.index(answer)],
    }


def main(
    dataset: str = "aqweteddy/Taiwan-Curlture-MCQ",
    url: str = "http://localhost:30001/v1",
    output_path: str = "output.json",
    use_gpt_as_parser: bool = False,
    lang: str = "zh",
):
    ds = load_dataset(dataset)["train"]
    # ds = ds.shuffle().select(range(100))
    ds = ds.map(rand_choices, num_proc=4)
    ds = ds.map(format_query, num_proc=4)
    questions = ds["_prompt"]
    cli = AsyncOpenAI(base_url=url)
    model_name = get_model_name(url)
    print(f"Evaluating model: {model_name}")
    answers = run_all(
        cli, questions, model_name, use_gpt_as_parser=use_gpt_as_parser, lang=lang
    )
    ds = ds.add_column("pred", answers)
    ds = ds.map(lambda x: {"_is_correct": x["pred"][-1]["content"] == x["answer"]})

    # overall accuracy
    correct = sum(ds["_is_correct"])
    total = len(ds)
    print(f"Overall accuracy: {correct}/{total} ({correct / total:.2%})")

    # group by accuracy
    grp_result = {}
    tot_grp_dct = Counter(ds["src"])
    cor_grp_dct = Counter()
    for x in ds:
        if x["_is_correct"]:
            cor_grp_dct[x["src"]] += 1

    for k, v in tot_grp_dct.items():
        cor = cor_grp_dct[k]
        print(f"{k}: ({cor / v:.2%})")
        grp_result[k] = cor / v

    print(grp_result)
    ds = ds.remove_columns([c for c in ds.column_names if c[0] == "_"])
    result = {
        "overall": correct / total,
        "group": grp_result,
        "detail": [x for x in ds],
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aqweteddy/Taiwan-Curlture-MCQ")
    parser.add_argument("--url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--use_gpt_as_parser", type=bool, default=True)
    args = parser.parse_args()
    main(**vars(args))
