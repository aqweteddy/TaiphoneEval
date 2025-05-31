import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import requests
from datasets import load_dataset
from litellm import acompletion, completion


def load_prompt(path: str):
    with open(path, "r") as f:
        prompts = [json.loads(line) for line in f]
    prompts_dct = {prompt["name"]: prompt for prompt in prompts}
    prompt_formatter_mapper = {}

    for prompt_name, prompt in prompts_dct.items():
        prompt_formatter_mapper[prompt_name] = (
            prompt["system_prompt"] + "\n\n" + prompt["prompt_template"]
        )

    prompt_output_mapper = {}
    for prompt_name, prompt in prompts_dct.items():
        prompt_output_mapper[prompt_name] = prompt["output_format"]

    return prompt_formatter_mapper, prompt_output_mapper


def map_prompt(
    dct: dict,
    prompt_formatter_mapper: dict,
    prompt_output_mapper: dict,
):
    if (
        len(dct["turns"]) == 1
        and dct["category"] not in ["math"]
        and not dct["reference"]
    ):
        formatter = "single-v1"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            question=dct["turns"][0],
            answer=dct["llm_response"],
        )
        prompt_output = prompt_output_mapper[formatter]

    elif (
        len(dct["turns"]) == 1 and dct["category"] not in ["math"] and dct["reference"]
    ):
        # formatter = "pair-v2"
        formatter = "single-v1"

        eval_prompt = prompt_formatter_mapper[formatter].format(
            question=dct["turns"][0],
            answer=dct["llm_response"],
        )
        prompt_output = prompt_output_mapper[formatter]

    elif (
        len(dct["turns"]) == 1 and dct["category"] in ["math"] and not dct["reference"]
    ):
        formatter = "single-math-v1"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            question=dct["turns"][0],
            answer=dct["llm_response"][0],
            ref_answer_1=dct["reference"][0],
        )
        prompt_output = prompt_output_mapper[formatter]
    elif len(dct["turns"]) == 1 and dct["category"] in ["math"] and dct["reference"]:
        formatter = "single-math-v1"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            question=dct["turns"][0],
            answer=dct["llm_response"][0],
            ref_answer_1=dct["reference"][0],
        )
    elif (
        len(dct["turns"]) == 2
        and dct["category"] not in ["math"]
        and not dct["reference"]
    ):
        formatter = "single-v1-multi-turn"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            question_1=dct["turns"][0],
            answer_1=dct["llm_response"][0],
            question_2=dct["turns"][1],
            answer_2=dct["llm_response"][1],
        )
        prompt_output = prompt_output_mapper[formatter]
    elif (
        len(dct["turns"]) == 2 and dct["category"] not in ["math"] and dct["reference"]
    ):
        # formatter = 'pair-v2-multi-turn'
        formatter = "single-v1-multi-turn"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            question_1=dct["turns"][0],
            answer_1=dct["llm_response"][0],
            question_2=dct["turns"][1],
            answer_2=dct["llm_response"][1],
        )
        prompt_output = prompt_output_mapper[formatter]
    elif (
        len(dct["turns"]) == 2 and dct["category"] in ["math"] and not dct["reference"]
    ):
        formatter = "single-math-v1-multi-turn"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            ref_answer_1=dct["reference"][0],
            question_1=dct["turns"][0],
            answer_1=dct["llm_response"][0],
            ref_answer_2=dct["reference"][1],
            question_2=dct["turns"][1],
            answer_2=dct["llm_response"][1],
        )
        prompt_output = prompt_output_mapper[formatter]
    elif len(dct["turns"]) == 2 and dct["category"] in ["math"] and dct["reference"]:
        formatter = "single-math-v1-multi-turn"
        eval_prompt = prompt_formatter_mapper[formatter].format(
            ref_answer_1=dct["reference"][0],
            question_1=dct["turns"][0],
            answer_1=dct["llm_response"][0],
            ref_answer_2=dct["reference"][1],
            question_2=dct["turns"][1],
            answer_2=dct["llm_response"][1],
        )
        prompt_output = prompt_output_mapper[formatter]
    return {
        "eval_prompt_type": formatter,
        "eval_prompt": eval_prompt,
        "eval_output_type": prompt_output,
    }


def gen_llm_response(
    dct: dict,
    url: str,
    model: str,
):
    # turn 1
    response = completion(
        model=model,
        messages=[{"role": "user", "content": dct["turns"][0]}],
        stream=False,
        base_url=url,
    )
    resp0 = response.choices[0].message.content

    # turn 2
    if len(dct["turns"]) == 2:
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": dct["turns"][0]},
                {"role": "assistant", "content": resp0},
                {"role": "user", "content": dct["turns"][1]},
            ],
            stream=False,
            base_url=url,
        )
        resp1 = response.choices[0].message.content

    return {
        "llm_response": [resp0, resp1] if len(dct["turns"]) == 2 else [resp0],
    }


def gen_judge_response(dct: dict, model: str, max_retries: int = 1):
    for _ in range(max_retries):
        try:
            judge_response = (
                completion(
                    model=model,
                    messages=[{"role": "user", "content": dct["eval_prompt"]}],
                    temperature=0.0,
                )
                .choices[0]
                .message.content
            )
            rating = float(judge_response.split("[[")[1].split("]]")[0])
            break
        except Exception as e:
            print(e)
            rating = None
    return {
        "score": rating,
        "reason": judge_response,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZoneTwelve/mt-bench-tw")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--eval_prompt_path", type=str, default="dict/mt_bench_prompt.jsonl"
    )

    args = parser.parse_args()

    ds = load_dataset(args.dataset)["train"]
    ds = ds.map(
        lambda x: {"reference": [] if x["reference"] == [""] else x["reference"]},
        num_proc=10,
    )

    if args.model is None:
        args.url = args.url.strip("/")
        args.model = "openai/" + requests.get(args.url + "/models").json()["data"][0]["id"]
        print(f"Using model: {args.model}")
    ds = ds.map(
        gen_llm_response,
        fn_kwargs={
            "url": args.url if args.url.lower() != "none" else None,
            "model": args.model,
        },
        num_proc=10,
    )

    prompt_formatter_mapper, prompt_output_mapper = load_prompt(args.eval_prompt_path)
    ds = ds.map(
        map_prompt,
        num_proc=10,
        fn_kwargs={
            "prompt_formatter_mapper": prompt_formatter_mapper,
            "prompt_output_mapper": prompt_output_mapper,
        },
    )
    ds = ds.map(
        gen_judge_response,
        fn_kwargs={
            "model": args.judge_model,
        },
        num_proc=10,
    )

    grp_metrics = defaultdict(lambda: [])

    metrics = {}
    for row in ds:
        if row["score"] is not None:
            grp_metrics[row["category"]].append(row["score"])

    for category, scores in grp_metrics.items():
        metrics[category] = {
            "mean": np.mean(scores).item(),
            "std": np.std(scores).item(),
        }
    print("overall metrics: ", metrics)
    with open(args.output_path, "w") as f:
        result = {
            "metrics": metrics,
            "results": ds.to_list(),
        }
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
