"""Extract structured legal facts from LeCaRDv2 queries and candidates."""

import json
import re
from pathlib import Path
from typing import Dict, List

from loguru import logger
from vllm import LLM, SamplingParams

from loaders import candidate_path, load_cases


root_path = Path(__file__).parent.parent.parent
data_path = root_path / "data" / "LeCaRDv2"


def load_queries():
    gear_query_file_path = data_path / "queries.jsonl"
    with gear_query_file_path.open("r", encoding="utf-8") as f:
        gear_queries = [json.loads(line) for line in f]
    gear_query_qids = [q["id"] for q in gear_queries]
    query_allcontext_path = data_path / "query_allcontext.json"
    queries = []
    with query_allcontext_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] in gear_query_qids:
                queries.append({"id": item["id"], "fact": item["fact"]})

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_candidates():
    candidates = load_cases(candidate_path)
    return [{"id": candidate["id"], "fact": candidate["fact"]} for candidate in candidates]


def extract_from_boxed_text(text: str) -> str:
    r"""Extract the content enclosed by <conclusion> tags."""
    pattern = r"<conclusion>(.*?)</conclusion>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No conclusion text found")


def get_fact_by_local_llm(model, template, cases: List[Dict]) -> Dict[int, str]:
    """Extract structured facts from cases with a local LLM."""
    prompts = []
    ids = []

    for case in cases:
        prompt = template.replace("$text$", case["fact"])
        ids.append(case["id"])
        prompts.append(prompt)

    message_list = [[{"role": "user", "content": p}] for p in prompts]

    responses = model.chat(
        message_list,
        sampling_params=SamplingParams(max_tokens=4096, temperature=0.01),  # pyright: ignore[reportArgumentType]
    )
    result_map = {}
    for cid, response in zip(ids, responses):
        text = response.outputs[0].text
        try:
            result = extract_from_boxed_text(text)
        except ValueError:
            logger.warning(f"Failed to parse case {cid}; LLM output: {text}")
            result = "[ERROR]"
        result_map[cid] = result
    return result_map


def extract_fact_from_cases(model, cases: List[Dict]) -> List[Dict]:
    """Extract the four structured criminal-fact views from cases."""
    subject_template = r"""请根据基本案情，抽取犯罪主体，并分析犯罪主体的构成要件。
犯罪主体的构成要件包括：是否达到法定年龄、是否是完全行为能力人。

<case>
$text$
</case>

先输出分析，最后结论使用
<conclusion>
...
</conclusion>
标注
"""
    subjective_element_template = r"""请根据基本案情，抽取犯罪主观方面，并说明犯罪意图。
犯罪主观方面包括：故意和过失。

<case>
$text$
</case>

先输出分析，最后结论使用
<conclusion>
...
</conclusion>
标注
"""
    object_template = r"""请根据基本案情，抽取犯罪客体，并给出犯罪客体的分析。
犯罪客体：刑法所保护而为犯罪所侵犯的社会关系。

<case>
$text$
</case>

先输出分析，最后结论使用
<conclusion>
...
</conclusion>
标注
"""
    object_element_template = r"""请根据基本案情，抽取犯罪的客观方面要素。
犯罪客观方面包括：犯罪时间、犯罪地点、犯罪行为、犯罪结果。

<case>
$text$
</case>

先输出分析，最后结论使用
<conclusion>
...
</conclusion>
标注
"""
    logger.info("Extracting criminal-subject information...")
    subject_info = get_fact_by_local_llm(model, subject_template, cases)
    logger.info("Extracting subjective-element information...")
    subjective_element_info = get_fact_by_local_llm(model, subjective_element_template, cases)
    logger.info("Extracting criminal-object information...")
    object_info = get_fact_by_local_llm(model, object_template, cases)
    logger.info("Extracting objective-element information...")
    object_element_info = get_fact_by_local_llm(model, object_element_template, cases)
    for c in cases:
        cid = c["id"]
        c["subject"] = subject_info[cid]
        c["subjective_element"] = subjective_element_info[cid]
        c["object"] = object_info[cid]
        c["objective_element"] = object_element_info[cid]
    return cases


def get_fact():
    """Extract structured facts for all cases and save them as JSONL files."""
    fact_dir = data_path / "LLM_extract"
    fact_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    candidates = load_candidates()

    model = LLM(model="path/to/Qwen/Qwen3-30B-A3B-Instruct-2507", tensor_parallel_size=4)
    logger.info("Extracting structured facts from queries...")
    queries = extract_fact_from_cases(model, queries)
    logger.info("Saving query facts to a JSONL file...")
    with (fact_dir / "query_fact.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info("Extracting structured facts from candidates...")
    candidates = extract_fact_from_cases(model, candidates)
    logger.info("Saving candidate facts to a JSONL file...")
    with (fact_dir / "candidate_fact.jsonl").open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    get_fact()
