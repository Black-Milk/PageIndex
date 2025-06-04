import asyncio
import copy
import json
import math
import os
import random
import re
from io import BytesIO

from . import utils


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):
    title = item["title"]
    if "physical_index" not in item or item["physical_index"] is None:
        return {
            "list_index": item.get("list_index"),
            "answer": "no",
            "title": title,
            "page_number": None,
        }

    page_number = item["physical_index"]
    page_text = page_list[page_number - start_index][0]

    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    Reply format:
    {{

        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await utils.ChatGPT_API_async(model=model, prompt=prompt)
    response = utils.extract_json(response)
    if "answer" in response:
        answer = response["answer"]
    else:
        answer = "no"
    return {
        "list_index": item["list_index"],
        "answer": answer,
        "title": title,
        "page_number": page_number,
    }


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await utils.ChatGPT_API_async(model=model, prompt=prompt)
    response = utils.extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(
    structure, page_list, model=None, logger=None
):
    if logger:
        logger.info("Checking title appearance in start concurrently")

    # skip items without physical_index
    for item in structure:
        if item.get("physical_index") is None:
            item["appear_start"] = "no"

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get("physical_index") is not None:
            page_text = page_list[item["physical_index"] - 1][0]
            tasks.append(
                check_title_appearance_in_start(
                    item["title"], page_text, model=model, logger=logger
                )
            )
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results, strict=False):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item["appear_start"] = "no"
        else:
            item["appear_start"] = result

    return structure


def toc_detector_single_page(content, model=None):
    """
    AI-POWERED SINGLE PAGE TOC DETECTOR

    Uses AI to determine if a given page contains table of contents content.
    This is the core detection engine used by find_toc_pages() to scan documents.

    Key Features:
    - Distinguishes TOCs from similar-looking content (abstracts, figure lists, etc.)
    - Handles various TOC formats and layouts
    - Returns simple "yes"/"no" classification

    AI Instructions:
    - Look for hierarchical list of sections/chapters
    - Exclude abstracts, summaries, notation lists, figure/table lists
    - Focus on document structure organization

    Used by: find_toc_pages() for progressive document scanning
    """
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = utils.ChatGPT_API(model=model, prompt=prompt)
    # print('response', response)
    json_content = utils.extract_json(response)
    return json_content["toc_detected"]


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = """
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + "\n Document:\n" + content + "\n Table of contents:\n" + toc
    response = utils.ChatGPT_API(model=model, prompt=prompt)
    json_content = utils.extract_json(response)
    return json_content["completed"]


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = """
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = (
        prompt
        + "\n Raw Table of contents:\n"
        + content
        + "\n Cleaned Table of contents:\n"
        + toc
    )
    response = utils.ChatGPT_API(model=model, prompt=prompt)
    json_content = utils.extract_json(response)
    return json_content["completed"]


def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = utils.ChatGPT_API_with_finish_reason(
        model=model, prompt=prompt
    )

    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response

    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    prompt = """please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = utils.ChatGPT_API_with_finish_reason(
        model=model, prompt=prompt, chat_history=chat_history
    )
    response = response + new_response
    if_complete = check_if_toc_transformation_is_complete(content, response, model)

    while not (if_complete == "yes" and finish_reason == "finished"):
        chat_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        prompt = """please continue the generation of table of contents , directly output the remaining part of the structure"""
        new_response, finish_reason = utils.ChatGPT_API_with_finish_reason(
            model=model, prompt=prompt, chat_history=chat_history
        )
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)

        # Optional: Add a maximum retry limit to prevent infinite loops
        if len(chat_history) > 5:  # Arbitrary limit of 10 attempts
            raise Exception(
                "Failed to complete table of contents after maximum retries"
            )

    return response


def detect_page_index(toc_content, model=None):
    """
    AI-POWERED PAGE NUMBER DETECTION

    Analyzes TOC content to determine if it includes page numbers or indices.
    This critical classification determines which processing strategy to use.

    Classification Logic:
    - "yes": TOC includes page numbers → Route to Strategy 1
    - "no": TOC lacks page numbers → Continue extended search or use Strategy 2

    Examples:
    "yes": "Chapter 1: Introduction ............ 15"
    "yes": "1. Background                        23"
    "no":  "Chapter 1: Introduction"
    "no":  "1. Background"

    Used by: toc_extractor() to classify discovered TOC content
    Impact: Directly influences strategy selection in check_toc()
    """
    print("start detect_page_index")
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = utils.ChatGPT_API(model=model, prompt=prompt)
    json_content = utils.extract_json(response)
    return json_content["page_index_given_in_toc"]


def toc_extractor(page_list, toc_page_list, model):
    """
    TOC CONTENT EXTRACTION & ANALYSIS COORDINATOR

    Takes the pages identified as containing TOC content and extracts the actual
    TOC text, then analyzes whether it includes page numbers. This function
    bridges the gap between page discovery and content analysis.

    Process:
    1. Extract raw text content from all TOC pages
    2. Clean and format the content (handle dots, spacing, etc.)
    3. Use AI to determine if the content includes page numbers
    4. Return structured result for strategy selection

    Text Cleaning:
    - Converts dot leaders (Chapter 1 ......... 15) to colons (Chapter 1: 15)
    - Handles various TOC formatting patterns
    - Normalizes spacing and punctuation

    Returns:
    {
        "toc_content": "Clean TOC text content",
        "page_index_given_in_toc": "yes" or "no"
    }

    Used by: check_toc() during initial analysis and extended search
    """

    def transform_dots_to_colon(text):
        # Convert dot leaders to colons for better AI processing
        text = re.sub(r"\.{5,}", ": ", text)
        # Handle dots separated by spaces
        text = re.sub(r"(?:\. ){5,}\.?", ": ", text)
        return text

    # STEP 1: EXTRACT RAW TOC CONTENT
    # Combine content from all pages identified as containing TOC
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]

    # STEP 2: CLEAN AND FORMAT CONTENT
    # Transform common TOC formatting patterns for better AI analysis
    toc_content = transform_dots_to_colon(toc_content)

    # STEP 3: ANALYZE FOR PAGE NUMBERS
    # Use AI to determine if this TOC includes page number information
    has_page_index = detect_page_index(toc_content, model=model)

    return {"toc_content": toc_content, "page_index_given_in_toc": has_page_index}


def toc_index_extractor(toc, content, model=None):
    print("start toc_index_extractor")
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = (
        tob_extractor_prompt
        + "\nTable of contents:\n"
        + str(toc)
        + "\nDocument pages:\n"
        + content
    )
    response = utils.ChatGPT_API(model=model, prompt=prompt)
    json_content = utils.extract_json(response)
    return json_content


def toc_transformer(toc_content, model=None):
    print("start toc_transformer")
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + "\n Given table of contents\n:" + toc_content
    last_complete, finish_reason = utils.ChatGPT_API_with_finish_reason(
        model=model, prompt=prompt
    )
    if_complete = check_if_toc_transformation_is_complete(
        toc_content, last_complete, model
    )
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = utils.extract_json(last_complete)
        cleaned_response = utils.convert_page_to_int(last_complete["table_of_contents"])
        return cleaned_response

    last_complete = utils.get_json_content(last_complete)
    while not (if_complete == "yes" and finish_reason == "finished"):
        position = last_complete.rfind("}")
        if position != -1:
            last_complete = last_complete[: position + 2]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = utils.ChatGPT_API_with_finish_reason(
            model=model, prompt=prompt
        )

        if new_complete.startswith("```json"):
            new_complete = utils.get_json_content(new_complete)
            last_complete = last_complete + new_complete

        if_complete = check_if_toc_transformation_is_complete(
            toc_content, last_complete, model
        )

    last_complete = json.loads(last_complete)

    cleaned_response = utils.convert_page_to_int(last_complete["table_of_contents"])
    return cleaned_response


def find_toc_pages(start_page_index, page_list, opt, logger=None):
    """
    TOC DISCOVERY ENGINE

    Scans through document pages to identify consecutive pages containing table of contents.
    Uses AI-based detection with smart stopping logic to efficiently find complete TOC sections.

    Args:
        start_page_index: Page to start scanning from (enables extended search)
        page_list: List of (page_content, token_count) tuples
        opt: Configuration options (includes toc_check_page_num limit)
        logger: Optional logging interface

    Returns:
        List of page indices that contain TOC content (e.g., [1, 2, 3] for multi-page TOC)

    Key Logic:
        - Scans pages sequentially using AI to detect TOC content
        - Stops immediately after finding complete TOC sequence (efficient!)
        - Respects search limits to avoid scanning entire large documents
        - Handles both single-page and multi-page TOCs
    """
    print("start find_toc_pages")

    # State tracking for smart stopping logic
    last_page_is_yes = False  # Track if previous page contained TOC
    toc_page_list = []  # Accumulate pages with TOC content
    i = start_page_index  # Current page being examined

    while i < len(page_list):
        # SMART STOPPING: Only check beyond limit if we're actively finding TOC pages
        # This prevents endless searching while allowing complete TOC discovery
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break

        # AI-BASED TOC DETECTION: Analyze current page content
        detected_result = toc_detector_single_page(page_list[i][0], model=opt.model)

        if detected_result == "yes":
            # FOUND TOC PAGE: Add to list and mark we're in a TOC sequence
            if logger:
                logger.info(f"Page {i} has toc")
            toc_page_list.append(i)
            last_page_is_yes = True

        elif detected_result == "no" and last_page_is_yes:
            # END OF TOC SEQUENCE: Previous page had TOC, this one doesn't
            # This means we've found the complete TOC section - stop searching!
            if logger:
                logger.info(f"Found the last page with toc: {i - 1}")
            break

        # CONTINUE SEARCHING: No TOC found yet, or not in a TOC sequence
        i += 1

    if not toc_page_list and logger:
        logger.info("No toc found")

    return toc_page_list


def remove_page_number(data):
    if isinstance(data, dict):
        data.pop("page_number", None)
        for key in list(data.keys()):
            if "nodes" in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get("title") == page_item.get("title"):
                physical_index = phy_item.get("physical_index")
                if (
                    physical_index is not None
                    and int(physical_index) >= start_page_index
                ):
                    pairs.append(
                        {
                            "title": phy_item.get("title"),
                            "page": page_item.get("page"),
                            "physical_index": physical_index,
                        }
                    )
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair["physical_index"]
            page_number = pair["page"]
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue

    if not differences:
        return None

    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1

    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]

    return most_common


def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get("page") is not None and isinstance(data[i]["page"], int):
            data[i]["physical_index"] = data[i]["page"] + offset
            del data[i]["page"]

    return data


def page_list_to_group_text(
    page_contents, token_lengths, max_tokens=20000, overlap_page=1
):
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]

    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(
        ((num_tokens / expected_parts_num) + max_tokens) / 2
    )

    for i, (page_content, page_tokens) in enumerate(
        zip(page_contents, token_lengths, strict=False)
    ):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append("".join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])

        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append("".join(current_subset))

    print("divide page_list to groups", len(subsets))
    return subsets


def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = (
        fill_prompt_seq
        + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    )
    current_json_raw = utils.ChatGPT_API(model=model, prompt=prompt)
    json_result = utils.extract_json(current_json_raw)

    for item in json_result:
        if "start" in item:
            del item["start"]
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r"<physical_index_\d+>.*?<physical_index_\d+>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), "", 1)
    return text


### add verify completeness
def generate_toc_continue(toc_content, part, model="gpt-4o-2024-11-20"):
    """
    AI-POWERED TOC STRUCTURE CONTINUATION

    Extends existing TOC structure by analyzing the next chunk of document content.
    This maintains hierarchical consistency while adding new sections discovered
    in the current chunk.

    Key Challenges:
    - Maintain consistent numbering across chunks
    - Properly nest subsections under correct parent sections
    - Handle section continuations that span multiple chunks
    - Preserve exact section titles from source text

    Inputs:
    - toc_content: Previously built TOC structure from earlier chunks
    - part: Current document chunk with page markers

    AI Instructions:
    - Continue the existing hierarchical numbering
    - Only add sections that START in the current chunk
    - Maintain parent-child relationships in structure
    - Extract exact titles without modification

    Example:
    Previous: [{"structure": "1", "title": "Introduction"}, {"structure": "1.1", "title": "Overview"}]
    Current chunk contains: "1.2 Methodology" and "2 Results"
    Output: [{"structure": "1.2", "title": "Methodology"}, {"structure": "2", "title": "Results"}]
    """
    print("start generate_toc_continue")

    # AI PROMPT: Instructs model to continue existing structure
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    prompt = (
        prompt
        + "\nGiven text\n:"
        + part
        + "\nPrevious tree structure\n:"
        + json.dumps(toc_content, indent=2)
    )
    response, finish_reason = utils.ChatGPT_API_with_finish_reason(
        model=model, prompt=prompt
    )
    if finish_reason == "finished":
        return utils.extract_json(response)
    else:
        raise Exception(f"finish reason: {finish_reason}")


### add verify completeness
def generate_toc_init(part, model=None):
    """
    AI-POWERED TOC STRUCTURE INITIALIZATION

    Analyzes the first chunk of a document to establish the foundational TOC structure.
    This is the starting point for Strategy 3 (No TOC) where we generate structure
    from scratch by examining document content.

    AI Task:
    - Identify section headings and their hierarchical relationships
    - Establish numbering system (1, 1.1, 1.2, 2, 2.1, etc.)
    - Extract exact section titles from document text
    - Determine physical page locations using page markers

    Input Format:
    <physical_index_1>
    [Page content...]
    <physical_index_1>

    <physical_index_2>
    [Page content...]
    <physical_index_2>

    Output Format:
    [
        {
            "structure": "1",
            "title": "Introduction",
            "physical_index": "<physical_index_5>"
        },
        {
            "structure": "1.1",
            "title": "Background",
            "physical_index": "<physical_index_7>"
        }
    ]
    """
    print("start generate_toc_init")

    # AI PROMPT: Instructs the model to extract hierarchical structure
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},

        ],


    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + "\nGiven text\n:" + part
    response, finish_reason = utils.ChatGPT_API_with_finish_reason(
        model=model, prompt=prompt
    )

    if finish_reason == "finished":
        return utils.extract_json(response)
    else:
        raise Exception(f"finish reason: {finish_reason}")


def process_no_toc(page_list, start_index=1, model=None, logger=None):
    """
    STRATEGY 3: GENERATE TOC FROM SCRATCH

    Used when no table of contents is found in the document. This strategy
    analyzes the entire document content to identify sections, subsections,
    and their hierarchical relationships, then generates a complete TOC structure.

    How it works:
    - Divides document into manageable chunks
    - Uses AI to identify section titles and hierarchy (1, 1.1, 1.2, 2, etc.)
    - Progressively builds TOC structure across document chunks
    - Maintains hierarchical numbering and proper nesting

    Advantages:
    - Works on any document, even those without TOCs
    - Creates standardized hierarchical structure
    - Identifies section boundaries and page locations

    Process:
    1. Prepare document chunks with page markers
    2. Generate initial TOC structure from first chunk
    3. Progressively extend structure through remaining chunks
    4. Maintain hierarchical consistency across chunks
    """

    page_contents = []
    token_lengths = []

    # STEP 1: PREPARE DOCUMENT CHUNKS WITH PAGE MARKERS
    # Add physical page markers to enable AI to track locations
    for page_index in range(start_index, start_index + len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index - start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(utils.count_tokens(page_text, model))

    # STEP 2: DIVIDE INTO TOKEN-LIMITED GROUPS
    # Ensure each chunk fits within AI model token limits (~20k tokens default)
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f"len(group_texts): {len(group_texts)}")

    # STEP 3: GENERATE INITIAL TOC STRUCTURE
    # Use AI to analyze first chunk and create foundational hierarchy
    toc_with_page_number = generate_toc_init(group_texts[0], model)

    # STEP 4: PROGRESSIVELY EXTEND TOC STRUCTURE
    # For each remaining chunk, continue building the hierarchy
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(
            toc_with_page_number, group_text, model
        )
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f"generate_toc: {toc_with_page_number}")

    # STEP 5: STANDARDIZE FORMAT
    # Convert AI-generated page markers to integer page numbers
    toc_with_page_number = utils.convert_physical_index_to_int(toc_with_page_number)
    logger.info(f"convert_physical_index_to_int: {toc_with_page_number}")

    return toc_with_page_number


def process_toc_no_page_numbers(
    toc_content, toc_page_list, page_list, start_index=1, model=None, logger=None
):
    """
    STRATEGY 2: PROCESS TOC WITHOUT PAGE NUMBERS

    Handles documents that have a table of contents but no page numbers listed.
    This strategy leverages the existing TOC structure (titles and hierarchy)
    and uses AI to match each section to its actual page location.

    Advantages over Strategy 3:
    - Preserves original TOC titles and hierarchy structure
    - More accurate than generating structure from scratch
    - Handles documents with complex or non-standard section titles

    Process:
    1. Parse existing TOC structure (titles only, no pages)
    2. Divide document into manageable chunks with page markers
    3. Use AI to progressively match TOC sections to document pages
    4. Build complete section-to-page mapping
    """

    page_contents = []
    token_lengths = []

    # STEP 1: PARSE EXISTING TOC STRUCTURE
    # Convert raw TOC text into structured JSON (titles and hierarchy only)
    toc_content = toc_transformer(toc_content, model)
    logger.info(f"toc_transformer: {toc_content}")

    # STEP 2: PREPARE DOCUMENT FOR AI ANALYSIS
    # Add page markers to enable AI to identify specific page locations
    for page_index in range(start_index, start_index + len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index - start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(utils.count_tokens(page_text, model))

    # STEP 3: DIVIDE INTO MANAGEABLE CHUNKS
    # Large documents need to be processed in groups to stay within AI token limits
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f"len(group_texts): {len(group_texts)}")

    # STEP 4: PROGRESSIVE SECTION-TO-PAGE MATCHING
    # Start with a copy of the parsed TOC structure
    toc_with_page_number = copy.deepcopy(toc_content)

    # For each document chunk, use AI to match TOC sections to pages
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(
            group_text, toc_with_page_number, model
        )
    logger.info(f"add_page_number_to_toc: {toc_with_page_number}")

    # STEP 5: CONVERT TO STANDARD FORMAT
    # Transform AI-generated page markers into integer page numbers
    toc_with_page_number = utils.convert_physical_index_to_int(toc_with_page_number)
    logger.info(f"convert_physical_index_to_int: {toc_with_page_number}")

    return toc_with_page_number


def process_toc_with_page_numbers(
    toc_content,
    toc_page_list,
    page_list,
    toc_check_page_num=None,
    model=None,
    logger=None,
):
    """
    STRATEGY 1: PROCESS TOC WITH PAGE NUMBERS

    Handles documents where the TOC includes page numbers, but those numbers may not
    match the actual PDF page indices. This strategy calculates the offset between
    TOC page numbers and physical page locations.

    Common scenarios:
    - TOC says "Chapter 1: Page 15" but it's actually on PDF page 17 (offset +2)
    - Document page numbering starts from 1, but PDF pages start from 0
    - Front matter (title, copyright, etc.) not counted in TOC page numbers

    Process:
    1. Parse TOC structure from raw text
    2. Use AI to find actual page locations for sample sections
    3. Calculate offset between TOC pages and physical pages
    4. Apply offset to all TOC entries
    5. Fill in any missing page numbers
    """

    # STEP 1: PARSE TOC STRUCTURE
    # Convert raw TOC text into structured JSON with titles and page numbers
    toc_with_page_number = toc_transformer(toc_content, model)
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    # STEP 2: CREATE VALIDATION COPY
    # Remove page numbers to create version for AI validation
    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))

    # STEP 3: SAMPLE DOCUMENT CONTENT FOR VALIDATION
    # Take pages after TOC to find where sections actually appear
    start_page_index = toc_page_list[-1] + 1  # Start after last TOC page
    main_content = ""

    # Build content with page markers for AI analysis
    for page_index in range(
        start_page_index, min(start_page_index + toc_check_page_num, len(page_list))
    ):
        main_content += f"<physical_index_{page_index + 1}>\n{page_list[page_index][0]}\n<physical_index_{page_index + 1}>\n\n"

    # STEP 4: AI-BASED PHYSICAL LOCATION DETECTION
    # Use AI to identify where each TOC section actually starts in the document
    toc_with_physical_index = toc_index_extractor(
        toc_no_page_number, main_content, model
    )
    logger.info(f"toc_with_physical_index: {toc_with_physical_index}")

    toc_with_physical_index = utils.convert_physical_index_to_int(
        toc_with_physical_index
    )
    logger.info(f"toc_with_physical_index: {toc_with_physical_index}")

    # STEP 5: CALCULATE PAGE OFFSET
    # Match TOC entries with their actual locations to find consistent offset
    matching_pairs = extract_matching_page_pairs(
        toc_with_page_number, toc_with_physical_index, start_page_index
    )
    logger.info(f"matching_pairs: {matching_pairs}")

    # Find the most common difference between TOC pages and actual pages
    offset = calculate_page_offset(matching_pairs)
    logger.info(f"offset: {offset}")

    # STEP 6: APPLY OFFSET TO ALL TOC ENTRIES
    # Adjust all TOC page numbers by the calculated offset
    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    # STEP 7: FILL MISSING PAGE NUMBERS
    # Some TOC entries might still lack page numbers - use AI to find them
    toc_with_page_number = process_none_page_numbers(
        toc_with_page_number, page_list, model=model
    )
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    return toc_with_page_number


##check if needed to process none page numbers
def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get("physical_index") is not None:
                    prev_physical_index = toc_items[j]["physical_index"]
                    break

            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get("physical_index") is not None:
                    next_physical_index = toc_items[j]["physical_index"]
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index + 1):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_index - start_index][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)

            item_copy = copy.deepcopy(item)
            del item_copy["page"]
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]["physical_index"], str) and result[0][
                "physical_index"
            ].startswith("<physical_index"):
                item["physical_index"] = int(
                    result[0]["physical_index"].split("_")[-1].rstrip(">").strip()
                )
                del item["page"]

    return toc_items


def check_toc(page_list, opt=None):
    """
    TOC INTELLIGENCE GATHERING & EXTENDED SEARCH ORCHESTRATOR

    This is the central decision-making function that determines which TOC processing
    strategy the system will use. It performs progressive TOC discovery with extended
    search capabilities to handle complex document layouts.

    Returns:
        Dictionary with TOC findings:
        - toc_content: Raw TOC text content (or None if no TOC)
        - toc_page_list: List of pages containing the selected TOC
        - page_index_given_in_toc: "yes"/"no" indicating if TOC has page numbers

    Processing Strategy Selection:
        - "yes" → Route to Strategy 1 (process_toc_with_page_numbers)
        - "no" with content → Route to Strategy 2 (process_toc_no_page_numbers)
        - None → Route to Strategy 3 (process_no_toc)
    """

    # PHASE 1: INITIAL TOC DISCOVERY
    # Search from beginning of document for TOC pages
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)

    if len(toc_page_list) == 0:
        # NO TOC FOUND: Route to Strategy 3 (generate TOC from scratch)
        print("no toc found")
        return {
            "toc_content": None,
            "toc_page_list": [],
            "page_index_given_in_toc": "no",
        }
    else:
        print("toc found")

        # PHASE 2: ANALYZE INITIAL TOC
        # Extract content and check if it includes page numbers
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json["page_index_given_in_toc"] == "yes":
            # SUCCESS: Found TOC with page numbers → Route to Strategy 1
            print("index found")
            return {
                "toc_content": toc_json["toc_content"],
                "toc_page_list": toc_page_list,
                "page_index_given_in_toc": "yes",
            }
        else:
            # PHASE 3: EXTENDED SEARCH FOR BETTER TOC
            # Initial TOC lacks page numbers - search for detailed TOC sections
            # Many documents have: Brief TOC (no pages) + Detailed TOC (with pages)

            current_start_index = toc_page_list[-1] + 1  # Start after initial TOC

            while (
                toc_json["page_index_given_in_toc"] == "no"  # Still no page numbers
                and current_start_index < len(page_list)  # Haven't reached end
                and current_start_index < opt.toc_check_page_num  # Within search limit
            ):
                # SEARCH FOR ADDITIONAL TOC SECTIONS
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index, page_list=page_list, opt=opt
                )

                if len(additional_toc_pages) == 0:
                    # No more TOC sections found - stop searching
                    break

                # ANALYZE NEW TOC SECTION
                additional_toc_json = toc_extractor(
                    page_list, additional_toc_pages, opt.model
                )

                if additional_toc_json["page_index_given_in_toc"] == "yes":
                    # SUCCESS: Found detailed TOC with page numbers!
                    # Replace initial TOC with this better version → Route to Strategy 1
                    print("index found")
                    return {
                        "toc_content": additional_toc_json["toc_content"],
                        "toc_page_list": additional_toc_pages,  # Use new TOC pages
                        "page_index_given_in_toc": "yes",
                    }
                else:
                    # This TOC section also lacks page numbers - continue searching
                    current_start_index = additional_toc_pages[-1] + 1

            # FALLBACK: Extended search found no TOC with page numbers
            # Return initial TOC without page numbers → Route to Strategy 2
            print("index not found")
            return {
                "toc_content": toc_json["toc_content"],
                "toc_page_list": toc_page_list,
                "page_index_given_in_toc": "no",
            }


################### fix incorrect toc #########################################################
def single_toc_item_index_fixer(section_title, content, model="gpt-4o-2024-11-20"):
    tob_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = (
        tob_extractor_prompt
        + "\nSection Title:\n"
        + str(section_title)
        + "\nDocument pages:\n"
        + content
    )
    response = utils.ChatGPT_API(model=model, prompt=prompt)
    json_content = utils.extract_json(response)
    return utils.convert_physical_index_to_int(json_content["physical_index"])


async def fix_incorrect_toc(
    toc_with_page_number,
    page_list,
    incorrect_results,
    start_index=1,
    model=None,
    logger=None,
):
    print(f"start fix_incorrect_toc with {len(incorrect_results)} incorrect results")
    incorrect_indices = {result["list_index"] for result in incorrect_results}

    end_index = len(page_list) + start_index - 1

    incorrect_results_and_range_logs = []

    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item["list_index"]
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index - 1, -1, -1):
            if i not in incorrect_indices:
                prev_correct = toc_with_page_number[i]["physical_index"]
                break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1

        # Find the next correct item
        next_correct = None
        for i in range(list_index + 1, len(toc_with_page_number)):
            if i not in incorrect_indices:
                next_correct = toc_with_page_number[i]["physical_index"]
                break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index

        incorrect_results_and_range_logs.append(
            {
                "list_index": list_index,
                "title": incorrect_item["title"],
                "prev_correct": prev_correct,
                "next_correct": next_correct,
            }
        )

        page_contents = []
        for page_index in range(prev_correct, next_correct + 1):
            page_text = f"<physical_index_{page_index}>\n{page_list[page_index - start_index][0]}\n<physical_index_{page_index}>\n\n"
            page_contents.append(page_text)
        content_range = "".join(page_contents)

        physical_index_int = single_toc_item_index_fixer(
            incorrect_item["title"], content_range, model
        )

        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item["physical_index"] = physical_index_int
        check_result = await check_title_appearance(
            check_item, page_list, start_index, model
        )

        return {
            "list_index": list_index,
            "title": incorrect_item["title"],
            "physical_index": physical_index_int,
            "is_valid": check_result["answer"] == "yes",
        }

    # Process incorrect items concurrently
    tasks = [process_and_check_item(item) for item in incorrect_results]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results, strict=False):
        if isinstance(result, Exception):
            print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result["is_valid"]:
            toc_with_page_number[result["list_index"]]["physical_index"] = result[
                "physical_index"
            ]
        else:
            invalid_results.append(
                {
                    "list_index": result["list_index"],
                    "title": result["title"],
                    "physical_index": result["physical_index"],
                }
            )

    logger.info(f"incorrect_results_and_range_logs: {incorrect_results_and_range_logs}")
    logger.info(f"invalid_results: {invalid_results}")

    return toc_with_page_number, invalid_results


async def fix_incorrect_toc_with_retries(
    toc_with_page_number,
    page_list,
    incorrect_results,
    start_index=1,
    max_attempts=3,
    model=None,
    logger=None,
):
    print("start fix_incorrect_toc")
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")

        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc, page_list, current_incorrect, start_index, model, logger
        )

        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break

    return current_toc, current_incorrect


################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    """
    TOC ACCURACY VERIFICATION SYSTEM

    Tests the quality of TOC results by verifying that section titles actually
    appear on their claimed pages. This is critical for catching errors in all
    three processing strategies.

    Verification Process:
    1. Sample TOC entries (all entries or random subset)
    2. For each entry, use AI to check if the title appears on the claimed page
    3. Calculate overall accuracy percentage
    4. Return list of incorrect entries for potential fixing

    Quality Thresholds:
    - 100% accuracy: Perfect result, use immediately
    - 60-99% accuracy: Good result, attempt to fix errors
    - <60% accuracy: Poor result, fall back to next strategy

    Args:
        N: Number of entries to test (None = test all entries)

    Returns:
        (accuracy_percentage, list_of_incorrect_entries)
    """
    print("start verify_toc")

    # EARLY EXIT: Check if we have viable results
    # Find the last valid page number to ensure we're processing meaningful content
    last_physical_index = None
    for item in reversed(list_result):
        if item.get("physical_index") is not None:
            last_physical_index = item["physical_index"]
            break

    # If TOC doesn't cover enough of the document, it's probably incomplete
    if last_physical_index is None or last_physical_index < len(page_list) / 2:
        return 0, []

    # STEP 1: DETERMINE VERIFICATION SCOPE
    if N is None:
        # TEST ALL ENTRIES: Most thorough verification
        print("check all items")
        sample_indices = range(0, len(list_result))
    else:
        # RANDOM SAMPLING: Faster verification for large TOCs
        N = min(N, len(list_result))
        print(f"check {N} items")
        sample_indices = random.sample(range(0, len(list_result)), N)

    # STEP 2: PREPARE VERIFICATION DATA
    # Add list indices to track which entries fail verification
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        item_with_index = item.copy()
        item_with_index["list_index"] = idx  # Track original position
        indexed_sample_list.append(item_with_index)

    # STEP 3: CONCURRENT VERIFICATION
    # Test all sampled entries in parallel for efficiency
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)

    # STEP 4: ANALYZE RESULTS
    correct_count = 0
    incorrect_results = []

    for result in results:
        if result["answer"] == "yes":
            # Title found on claimed page - correct!
            correct_count += 1
        else:
            # Title NOT found on claimed page - needs fixing
            incorrect_results.append(result)

    # STEP 5: CALCULATE ACCURACY METRICS
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy * 100:.2f}%")

    return accuracy, incorrect_results


################### main process #########################################################
async def meta_processor(
    page_list,
    mode=None,
    toc_content=None,
    toc_page_list=None,
    start_index=1,
    opt=None,
    logger=None,
):
    """
    IMPROVED STRATEGY EXECUTION ENGINE & FALLBACK ROUTER

    Now with simplified fallback logic and better error handling.

    Simplified Fallback Chain:
    - Strategy 1 → Strategy 2 (logical: use TOC structure without page numbers)
    - Strategy 2 → Strategy 3 (fallback: generate from scratch)
    - Strategy 3 → Exception (no more options)

    Improvements:
    - Clearer fallback logic
    - Better logging and error reporting
    - More targeted fallback decisions
    """
    logger.info(f"Executing strategy: {mode}")
    logger.info(f"Starting from page index: {start_index}")

    # STEP 1: EXECUTE CHOSEN STRATEGY
    try:
        if mode == "process_toc_with_page_numbers":
            # STRATEGY 1: TOC with page numbers (offset calculation)
            toc_with_page_number = process_toc_with_page_numbers(
                toc_content,
                toc_page_list,
                page_list,
                toc_check_page_num=opt.toc_check_page_num,
                model=opt.model,
                logger=logger,
            )
        elif mode == "process_toc_no_page_numbers":
            # STRATEGY 2: TOC without page numbers (AI matching)
            toc_with_page_number = process_toc_no_page_numbers(
                toc_content, toc_page_list, page_list, model=opt.model, logger=logger
            )
        else:
            # STRATEGY 3: No TOC (generate from scratch)
            toc_with_page_number = process_no_toc(
                page_list, start_index=start_index, model=opt.model, logger=logger
            )
    except Exception as e:
        if logger:
            logger.error(f"Strategy {mode} failed with error: {e}")
        # Immediate fallback on execution error
        return await _execute_fallback_strategy(
            mode, page_list, toc_content, toc_page_list, start_index, opt, logger
        )

    # STEP 2: CLEAN RESULTS
    # Remove any entries that lack page number mappings
    valid_results = [
        item for item in toc_with_page_number if item.get("physical_index") is not None
    ]

    if not valid_results:
        if logger:
            logger.warning(f"Strategy {mode} produced no valid results")
        return await _execute_fallback_strategy(
            mode, page_list, toc_content, toc_page_list, start_index, opt, logger
        )

    # STEP 3: VERIFY RESULT ACCURACY
    # Test if section titles actually appear on their claimed pages
    accuracy, incorrect_results = await verify_toc(
        page_list, valid_results, start_index=start_index, model=opt.model
    )

    if logger:
        logger.info(
            f"Strategy {mode} achieved {accuracy * 100:.1f}% accuracy with {len(incorrect_results)} errors"
        )

    # STEP 4: QUALITY-BASED DECISION MAKING
    if accuracy == 1.0 and len(incorrect_results) == 0:
        # PERFECT RESULT: Return immediately
        if logger:
            logger.info(f"Strategy {mode} succeeded with perfect accuracy")
        return valid_results

    if accuracy >= 0.6 and len(incorrect_results) > 0:
        # GOOD RESULT WITH FIXABLE ERRORS: Attempt corrections
        if logger:
            logger.info(
                f"Strategy {mode} has good accuracy, attempting to fix {len(incorrect_results)} errors"
            )
        try:
            fixed_results, remaining_errors = await fix_incorrect_toc_with_retries(
                valid_results,
                page_list,
                incorrect_results,
                start_index=start_index,
                max_attempts=3,
                model=opt.model,
                logger=logger,
            )
            if logger:
                logger.info(f"Fixed errors, {len(remaining_errors)} errors remain")
            return fixed_results
        except Exception as e:
            if logger:
                logger.error(f"Error fixing failed: {e}")
            # Continue to fallback if fixing fails

    # POOR RESULT: Implement fallback strategy
    if logger:
        logger.warning(
            f"Strategy {mode} failed with {accuracy * 100:.1f}% accuracy, falling back"
        )
    return await _execute_fallback_strategy(
        mode, page_list, toc_content, toc_page_list, start_index, opt, logger
    )


async def _execute_fallback_strategy(
    failed_mode, page_list, toc_content, toc_page_list, start_index, opt, logger
):
    """
    IMPROVED FALLBACK EXECUTION

    Implements logical fallback chain with clear reasoning.
    """
    if failed_mode == "process_toc_with_page_numbers":
        # Strategy 1 failed → Try Strategy 2 (logical: same TOC, different approach)
        if logger:
            logger.info("Fallback: Strategy 1 → Strategy 2 (ignoring page numbers)")
        return await meta_processor(
            page_list,
            mode="process_toc_no_page_numbers",
            toc_content=toc_content,
            toc_page_list=toc_page_list,
            start_index=start_index,
            opt=opt,
            logger=logger,
        )
    elif failed_mode == "process_toc_no_page_numbers":
        # Strategy 2 failed → Try Strategy 3 (fallback: abandon TOC, generate fresh)
        if logger:
            logger.info("Fallback: Strategy 2 → Strategy 3 (generating from scratch)")
        return await meta_processor(
            page_list,
            mode="process_no_toc",
            start_index=start_index,
            opt=opt,
            logger=logger,
        )
    else:
        # Strategy 3 failed → No more fallbacks available
        if logger:
            logger.error("All strategies failed - no more fallbacks available")
        raise Exception(
            f"All TOC processing strategies failed. Last attempt was: {failed_mode}"
        )


async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node["start_index"] - 1 : node["end_index"]]
    token_num = sum([page[1] for page in node_page_list])

    if (
        node["end_index"] - node["start_index"] > opt.max_page_num_each_node
        and token_num >= opt.max_token_num_each_node
    ):
        print(
            "large node:",
            node["title"],
            "start_index:",
            node["start_index"],
            "end_index:",
            node["end_index"],
            "token_num:",
            token_num,
        )

        node_toc_tree = await meta_processor(
            node_page_list,
            mode="process_no_toc",
            start_index=node["start_index"],
            opt=opt,
            logger=logger,
        )
        node_toc_tree = await check_title_appearance_in_start_concurrent(
            node_toc_tree, page_list, model=opt.model, logger=logger
        )

        if node["title"].strip() == node_toc_tree[0]["title"].strip():
            node["nodes"] = utils.post_processing(node_toc_tree[1:], node["end_index"])
            node["end_index"] = node_toc_tree[1]["start_index"]
        else:
            node["nodes"] = utils.post_processing(node_toc_tree, node["end_index"])
            node["end_index"] = node_toc_tree[0]["start_index"]

    if "nodes" in node and node["nodes"]:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node["nodes"]
        ]
        await asyncio.gather(*tasks)

    return node


def determine_optimal_strategy(check_toc_result, logger=None):
    """
    STRATEGY DECISION MATRIX

    Analyzes TOC discovery results and determines the optimal processing strategy.
    This centralizes all routing logic in one clear, testable function.

    Decision Rules:
    1. TOC found + has page numbers → Strategy 1 (process_toc_with_page_numbers)
    2. TOC found + no page numbers → Strategy 2 (process_toc_no_page_numbers)
    3. No TOC found → Strategy 3 (process_no_toc)

    Returns:
        dict: {
            "strategy": str,
            "reason": str,
            "toc_content": str|None,
            "toc_page_list": list
        }
    """
    has_toc_content = (
        check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip()
    )
    has_page_numbers = check_toc_result.get("page_index_given_in_toc") == "yes"

    if has_toc_content and has_page_numbers:
        # STRATEGY 1: Optimal case - TOC with page numbers
        decision = {
            "strategy": "process_toc_with_page_numbers",
            "reason": "Found TOC with page numbers - optimal processing path",
            "toc_content": check_toc_result["toc_content"],
            "toc_page_list": check_toc_result["toc_page_list"],
        }

    elif has_toc_content and not has_page_numbers:
        # STRATEGY 2: Good case - TOC structure exists, add page numbers via AI
        decision = {
            "strategy": "process_toc_no_page_numbers",
            "reason": "Found TOC without page numbers - will use AI to map sections to pages",
            "toc_content": check_toc_result["toc_content"],
            "toc_page_list": check_toc_result["toc_page_list"],
        }

    else:
        # STRATEGY 3: Fallback case - no TOC, generate from document analysis
        decision = {
            "strategy": "process_no_toc",
            "reason": "No TOC found - will generate structure from document content",
            "toc_content": None,
            "toc_page_list": [],
        }

    if logger:
        logger.info(f"Strategy Decision: {decision['strategy']} - {decision['reason']}")

    return decision


async def tree_parser(page_list, opt, doc=None, logger=None):
    """
    IMPROVED TOC PROCESSING ORCHESTRATOR

    Now uses explicit strategy decision matrix for clear, maintainable routing.
    """
    # STEP 1: GATHER TOC INTELLIGENCE
    check_toc_result = check_toc(page_list, opt)
    if logger:
        logger.info(f"TOC Analysis Result: {check_toc_result}")

    # STEP 2: DETERMINE OPTIMAL STRATEGY USING DECISION MATRIX
    strategy_decision = determine_optimal_strategy(check_toc_result, logger)

    # STEP 3: EXECUTE CHOSEN STRATEGY
    toc_with_page_number = await meta_processor(
        page_list,
        mode=strategy_decision["strategy"],
        start_index=1,
        toc_content=strategy_decision["toc_content"],
        toc_page_list=strategy_decision["toc_page_list"],
        opt=opt,
        logger=logger,
    )

    # STEP 4: POST-PROCESSING (unchanged)
    toc_with_page_number = utils.add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, page_list, model=opt.model, logger=logger
    )
    toc_tree = utils.post_processing(toc_with_page_number, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)

    return toc_tree


def page_index_main(doc, opt=None):
    logger = utils.JsonLogger(doc)

    is_valid_pdf = (
        isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")
    ) or isinstance(doc, BytesIO)
    if not is_valid_pdf:
        raise ValueError(
            "Unsupported input type. Expected a PDF file path or BytesIO object."
        )

    print("Parsing PDF...")
    page_list = utils.get_page_tokens(doc)

    logger.info({"total_page_number": len(page_list)})
    logger.info({"total_token": sum([page[1] for page in page_list])})

    structure = asyncio.run(tree_parser(page_list, opt, doc=doc, logger=logger))
    if opt.if_add_node_id == "yes":
        utils.write_node_id(structure)
    if opt.if_add_node_text == "yes":
        utils.add_node_text(structure, page_list)
    if opt.if_add_node_summary == "yes":
        if opt.if_add_node_text == "no":
            utils.add_node_text(structure, page_list)
        asyncio.run(utils.generate_summaries_for_structure(structure, model=opt.model))
        if opt.if_add_node_text == "no":
            utils.remove_structure_text(structure)
        if opt.if_add_doc_description == "yes":
            doc_description = utils.generate_doc_description(structure, model=opt.model)
            return {
                "doc_name": utils.get_pdf_name(doc),
                "doc_description": doc_description,
                "structure": structure,
            }
    return {
        "doc_name": utils.get_pdf_name(doc),
        "structure": structure,
    }


def page_index(
    doc,
    model=None,
    toc_check_page_num=None,
    max_page_num_each_node=None,
    max_token_num_each_node=None,
    if_add_node_id=None,
    if_add_node_summary=None,
    if_add_doc_description=None,
    if_add_node_text=None,
):
    user_opt = {
        arg: value
        for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = utils.ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)
