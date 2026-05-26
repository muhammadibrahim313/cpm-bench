"""
CPM-Bench Scale Pipeline: Multilingual Token Efficiency with Input + Output Cost
===============================================================================

This is the scale-ready version for Kaggle/offline runs.

What it does:
1. Builds a clean deterministic English prompt dataset across many categories.
2. Creates a short English reference output/answer for every prompt.
3. Translates prompts into target languages with NLLB.
4. Back-translates prompts into English for semantic quality checking.
5. Translates the reference output/answer into target languages.
6. Computes LaBSE semantic similarity for prompt back-translation quality.
7. Counts input tokens, output tokens, and total tokens with Qwen3 and NLLB tokenizers.
8. Computes:
   - input TER vs English
   - output TER vs English
   - total TER vs English
   - input/output token share
9. Saves master CSVs, summaries, review sheets, charts, and status JSON.
10. Uses low logging and safe checkpoints so Kaggle does not fail from huge notebook output.

Required Kaggle model inputs:
- /kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1
- /kaggle/input/models/bankhv/facebooknllb-200-distilled-1-3b/pytorch/v1/1/facebook/nllb-200-distilled-1.3B
- /kaggle/input/models/google/labse/tensorflow2/labse/2
- /kaggle/input/models/google/universal-sentence-encoder/tensorflow2/cmlm-multilingual-preprocess/2

Main files:
- /kaggle/working/token_efficiency_outputs/00_DATASET_MAIN_NO_TOKENIZER_DUPLICATES.csv
- /kaggle/working/token_efficiency_outputs/00_MASTER_FOR_REVIEW.csv
- /kaggle/working/token_efficiency_outputs/05B_overall_language_summary.csv

Recommended run order:
1. RUN_PRESET = "quick_5min"
2. RUN_PRESET = "v1_40cat_20lang_batch_a"
3. RUN_PRESET = "v1_40cat_20lang_batch_b"
4. Merge batch outputs with the companion merge script

Notes:
- For long runs, keep make_zip_bundle=False to avoid doubling disk usage.
- This script deletes old output_dir only when clean_output_dir_on_start=True.
- It avoids huge prints and saves checkpoint CSVs every N translation jobs.
"""

# ============================================================
# CHANGE THIS LINE ONLY MOST OF THE TIME
# ============================================================

# RUN_PRESET = "quick_5min"
# RUN_PRESET = "v1_40cat_20lang_batch_a"
# RUN_PRESET = "v1_40cat_20lang_batch_b"
RUN_PRESET = "v1_40cat_20lang_single_75"
# RUN_PRESET = "scale_30_categories_input_output"


# ============================================================
# IMPORTS AND QUIET SETTINGS
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

import re
import gc
import json
import time
import math
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# CONFIG
# ============================================================

BASE_CONFIG = {
    "run_name": "cpm_bench_scale_input_output",

    "qwen_model_path": "/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1",
    "nllb_model_path": "/kaggle/input/models/bankhv/facebooknllb-200-distilled-1-3b/pytorch/v1/1/facebook/nllb-200-distilled-1.3B",
    "labse_encoder_path": "/kaggle/input/models/google/labse/tensorflow2/labse/2",
    "labse_preprocessor_path": "/kaggle/input/models/google/universal-sentence-encoder/tensorflow2/cmlm-multilingual-preprocess/2",

    "output_dir": "/kaggle/working/token_efficiency_outputs",

    "target_languages": ["Urdu", "Arabic", "Hindi", "Chinese", "French"],
    "similarity_threshold": 0.85,

    "translation_max_new_tokens": 256,
    "translation_num_beams": 4,
    "translation_checkpoint_every": 250,
    "progress_print_every": 250,
    "resume_translations": True,

    "labse_batch_size": 128,

    "tokenizers_to_use": ["qwen3_14b_base", "nllb_200_distilled_1_3b"],

    # Output-side analysis. This translates English reference outputs into target languages
    # and counts output tokens. It is much faster and more stable than generating answers
    # from a large LLM for every prompt-language pair.
    "include_output_side_analysis": True,
    "translate_reference_outputs": True,
    "back_translate_outputs": False,

    "save_charts": True,
    "make_zip_bundle": False,
    "clean_output_dir_on_start": False,

    # Keep CSVs only; avoid notebook-heavy outputs.
    "print_dataframe_previews": False,
}

NLLB_LANG_CODES = {
    "English": "eng_Latn",
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "French": "fra_Latn",
    "Hindi": "hin_Deva",
    "Urdu": "urd_Arab",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Japanese": "jpn_Jpan",
    "Bengali": "ben_Beng",
    "Punjabi": "pan_Guru",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Korean": "kor_Hang",
    "Vietnamese": "vie_Latn",
    "Turkish": "tur_Latn",
    "Indonesian": "ind_Latn",
    "Persian": "pes_Arab",
}

TARGET_LANGUAGES_20 = [
    "Arabic", "Chinese", "French", "Hindi", "Urdu",
    "Spanish", "German", "Portuguese", "Russian", "Japanese",
    "Bengali", "Punjabi", "Tamil", "Telugu", "Marathi",
    "Korean", "Vietnamese", "Turkish", "Indonesian", "Persian",
]

CATEGORIES_40 = [
    "science_factual_qa", "science_reasoning", "history_factual_qa", "geography_factual_qa",
    "math_word_problem", "summarization", "classification", "sentiment_classification",
    "information_extraction", "named_entity_extraction", "structured_json", "table_conversion",
    "code_generation", "code_explanation", "debugging_help", "professional_writing",
    "email_writing", "customer_support", "plain_language_legal", "plain_language_medical",
    "plain_language_finance", "education_tutoring", "creative_writing", "safety_explanation",
    "comparison_question", "planning_question", "instruction_following", "data_analysis_question",
    "cultural_localization", "translation_instruction", "fact_checking", "argument_analysis",
    "decision_making", "project_management", "meeting_notes", "policy_explanation",
    "research_question", "data_cleaning_instruction", "algorithmic_reasoning", "multilingual_localization_advice",
]

CATEGORIES_40_BATCH_A = CATEGORIES_40[:20]
CATEGORIES_40_BATCH_B = CATEGORIES_40[20:]



PRESETS = {
    "quick_5min": {
        "run_name": "quick_5min_input_output_test",
        "categories_to_run": [
            "science_factual_qa",
            "science_reasoning",
            "history_factual_qa",
        ],
        "num_prompts_per_category": 2,
        "target_languages": ["Urdu", "Chinese"],
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "translation_checkpoint_every": 25,
        "progress_print_every": 10,
        "make_zip_bundle": True,
    },
    "v1_40cat_20lang_batch_a": {
        "run_name": "v1_40cat_20lang_batch_a_100_prompts_input_output",
        "categories_to_run": CATEGORIES_40_BATCH_A,
        "num_prompts_per_category": 100,
        "target_languages": TARGET_LANGUAGES_20,
        "output_dir": "/kaggle/working/cpm_bench_v1_40x20_batch_A",
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 1000,
        "translation_checkpoint_every": 1000,
        "labse_batch_size": 128,
        "make_zip_bundle": False,
    },
    "v1_40cat_20lang_batch_b": {
        "run_name": "v1_40cat_20lang_batch_b_100_prompts_input_output",
        "categories_to_run": CATEGORIES_40_BATCH_B,
        "num_prompts_per_category": 100,
        "target_languages": TARGET_LANGUAGES_20,
        "output_dir": "/kaggle/working/cpm_bench_v1_40x20_batch_B",
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 1000,
        "translation_checkpoint_every": 1000,
        "labse_batch_size": 128,
        "make_zip_bundle": False,
    },
    "v1_40cat_20lang_single_75": {
        "run_name": "v1_40cat_20lang_single_75_prompts_input_output",
        "categories_to_run": CATEGORIES_40,
        "num_prompts_per_category": 75,
        "target_languages": TARGET_LANGUAGES_20,
        "output_dir": "/kaggle/working/cpm_bench_v1_40x20_single_75",
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 1000,
        "translation_checkpoint_every": 1000,
        "labse_batch_size": 128,
        "make_zip_bundle": False,
    },
    "scale_10_categories": {
        "run_name": "scale_10_categories_100_prompts_input_output",
        "categories_to_run": [
            "science_factual_qa",
            "science_reasoning",
            "history_factual_qa",
            "math_word_problem",
            "summarization",
            "classification",
            "information_extraction",
            "structured_json",
            "code_generation",
            "professional_writing",
        ],
        "num_prompts_per_category": 100,
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 250,
        "translation_checkpoint_every": 250,
        "make_zip_bundle": False,
    },
    "scale_30_categories_input_output": {
        "run_name": "scale_30_categories_100_prompts_input_output",
        "categories_to_run": [
            "science_factual_qa",
            "science_reasoning",
            "history_factual_qa",
            "geography_factual_qa",
            "math_word_problem",
            "summarization",
            "classification",
            "sentiment_classification",
            "information_extraction",
            "named_entity_extraction",
            "structured_json",
            "table_conversion",
            "code_generation",
            "code_explanation",
            "debugging_help",
            "professional_writing",
            "email_writing",
            "customer_support",
            "plain_language_legal",
            "plain_language_medical",
            "plain_language_finance",
            "education_tutoring",
            "creative_writing",
            "safety_explanation",
            "comparison_question",
            "planning_question",
            "instruction_following",
            "data_analysis_question",
            "cultural_localization",
            "translation_instruction",
        ],
        "num_prompts_per_category": 100,
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 500,
        "translation_checkpoint_every": 500,
        "make_zip_bundle": False,
    },
    "scale_30_categories_fast_no_output": {
        "run_name": "scale_30_categories_100_prompts_input_only_fast",
        "categories_to_run": [
            "science_factual_qa",
            "science_reasoning",
            "history_factual_qa",
            "geography_factual_qa",
            "math_word_problem",
            "summarization",
            "classification",
            "sentiment_classification",
            "information_extraction",
            "named_entity_extraction",
            "structured_json",
            "table_conversion",
            "code_generation",
            "code_explanation",
            "debugging_help",
            "professional_writing",
            "email_writing",
            "customer_support",
            "plain_language_legal",
            "plain_language_medical",
            "plain_language_finance",
            "education_tutoring",
            "creative_writing",
            "safety_explanation",
            "comparison_question",
            "planning_question",
            "instruction_following",
            "data_analysis_question",
            "cultural_localization",
            "translation_instruction",
        ],
        "num_prompts_per_category": 100,
        "include_output_side_analysis": False,
        "translate_reference_outputs": False,
        "clean_output_dir_on_start": True,
        "resume_translations": False,
        "progress_print_every": 500,
        "translation_checkpoint_every": 500,
        "make_zip_bundle": False,
    },
}

if RUN_PRESET not in PRESETS:
    raise ValueError(f"Unknown RUN_PRESET: {RUN_PRESET}")

CONFIG = BASE_CONFIG.copy()
CONFIG.update(PRESETS[RUN_PRESET])

# ============================================================
# CATEGORY SPECIFICATIONS
# ============================================================

def common_pools() -> Dict[str, List[str]]:
    return {
        "science_concept": ["photosynthesis", "gravity", "DNA", "the water cycle", "evaporation", "vaccination", "electric current", "friction", "the ozone layer", "climate"],
        "history_event": ["the Renaissance", "the Industrial Revolution", "World War II", "the Cold War", "the French Revolution", "the moon landing", "the Silk Road", "the Roman Empire", "the United Nations", "the printing press"],
        "place": ["Japan", "Brazil", "Egypt", "Canada", "Pakistan", "India", "France", "China", "South Africa", "Germany"],
        "city": ["Tokyo", "Cairo", "Paris", "Lahore", "Beijing", "Berlin", "Toronto", "Delhi", "Karachi", "Cape Town"],
        "topic": ["renewable energy", "online learning", "public transport", "clean water", "digital privacy", "healthy sleep", "team communication", "urban gardening", "space exploration", "climate adaptation"],
        "product": ["laptop", "mobile phone", "water bottle", "backpack", "headphones", "desk lamp", "notebook", "router", "printer", "keyboard"],
        "service": ["internet service", "online delivery", "banking app", "university portal", "hotel booking", "mobile data plan", "public library", "training workshop", "customer support chat", "cloud storage"],
        "audience": ["high-school students", "new employees", "busy parents", "small business owners", "first-year university students", "software developers", "teachers", "healthcare staff", "travelers", "research assistants"],
        "tone": ["polite", "professional", "friendly", "concise", "formal", "supportive", "clear", "calm", "encouraging", "neutral"],
        "language_a": ["English", "Urdu", "Arabic", "Hindi", "Chinese", "French", "Spanish", "German"],
        "language_b": ["Urdu", "Arabic", "Hindi", "Chinese", "French", "English", "German", "Spanish"],
        "claim": ["online learning improves access", "public transport reduces traffic", "renewable energy is always cheaper", "social media improves communication", "remote work increases productivity", "exercise improves focus", "privacy policies are easy to understand", "AI tools save time", "electric cars reduce emissions", "homework improves learning"],
        "option_a": ["online classes", "public transport", "solar energy", "cloud storage", "team chat", "Python", "manual review", "short summaries", "open-source tools", "mobile banking"],
        "option_b": ["in-person classes", "private cars", "wind energy", "local storage", "email", "JavaScript", "automatic filtering", "detailed reports", "commercial tools", "branch banking"],
        "dataset_issue": ["missing values", "duplicate rows", "inconsistent date formats", "extra spaces", "mixed language text", "incorrect labels", "outlier values", "empty columns", "encoding errors", "mismatched IDs"],
        "policy_topic": ["data privacy", "refund requests", "remote work", "attendance", "safe password use", "AI tool usage", "customer complaints", "document sharing", "online exams", "device security"],
        "meeting_topic": ["dataset review", "project planning", "translation quality", "model evaluation", "customer feedback", "research progress", "dashboard design", "bug triage", "release planning", "team onboarding"],
    }

POOLS = common_pools()

# Prompt variation pools are used to safely create many unique prompts from a small
# number of category templates. They avoid broken/incomplete prompts and prevent
# duplicate filtering from stopping large runs.
PROMPT_VARIANT_PREFIXES = [
    "",
    "In simple terms, ",
    "For a beginner, ",
    "For a classroom discussion, ",
    "Using clear language, ",
    "For a short answer, ",
    "For a high-school student, ",
    "In a practical context, ",
    "For someone new to the topic, ",
    "With a real-world focus, ",
]

PROMPT_VARIANT_SUFFIXES = [
    "",
    " Keep the answer concise.",
    " Use one simple example if helpful.",
    " Focus on the main idea.",
    " Avoid unnecessary technical detail.",
    " Make the response easy to understand.",
    " Give a clear and direct answer.",
    " Use plain language.",
    " Mention the most important point first.",
    " Keep the explanation practical.",
]

CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "science_factual_qa": {
        "subject": "Science", "category": "Factual QA", "subcategory": "basic_science_facts",
        "templates": [
            "What is {science_concept}, and why is it important?",
            "How does {science_concept} work in simple terms?",
            "Why do scientists study {science_concept}?",
            "What role does {science_concept} play in everyday life?",
        ],
        "answer_templates": ["A good answer should define {science_concept}, explain its main role, and mention why it matters in real life."],
    },
    "science_reasoning": {
        "subject": "Science", "category": "Reasoning", "subcategory": "basic_science_reasoning",
        "templates": [
            "If a plant receives less sunlight for several days, what will likely happen and why?",
            "Why does metal feel colder than wood at the same room temperature?",
            "If more force is applied to an object, how does its motion usually change?",
            "Why do we see lightning before hearing thunder?",
        ],
        "answer_templates": ["A good answer should identify the cause, describe the effect, and connect the explanation to a basic scientific principle."],
    },
    "history_factual_qa": {
        "subject": "History", "category": "Factual QA", "subcategory": "world_history_basic",
        "templates": [
            "What was {history_event}, and why is it historically important?",
            "How did {history_event} affect society?",
            "What is one major outcome of {history_event}?",
            "Why do historians still study {history_event}?",
        ],
        "answer_templates": ["A good answer should briefly describe {history_event}, explain its context, and mention one major historical impact."],
    },
    "geography_factual_qa": {
        "subject": "Geography", "category": "Factual QA", "subcategory": "world_geography_basic",
        "templates": [
            "Where is {place} located, and what is it known for?",
            "What is one important geographic feature of {place}?",
            "How can geography influence daily life in {place}?",
            "Why might people visit {city}?",
        ],
        "answer_templates": ["A good answer should identify the location, mention a relevant geographic feature, and explain why it matters."],
    },
    "math_word_problem": {
        "subject": "Mathematics", "category": "Reasoning", "subcategory": "word_problem",
        "templates": [
            "A shop sells 3 notebooks for 45 dollars. What is the price of one notebook?",
            "A train travels 120 kilometers in 2 hours. What is its average speed?",
            "A class has 18 girls and 12 boys. What percentage of the class are girls?",
            "A worker saves 25 dollars each week. How much will they save in 8 weeks?",
        ],
        "answer_templates": ["A good answer should identify the relevant numbers, perform the calculation step by step, and state the final result clearly."],
    },
    "summarization": {
        "subject": "General", "category": "Summarization", "subcategory": "short_summary",
        "templates": [
            "Summarize this idea in two sentences: {topic} can affect communities in many different ways.",
            "Summarize the main benefit of {topic} for {audience}.",
            "Write a short summary explaining why {topic} is important today.",
            "Summarize the following topic for a beginner: {topic}.",
        ],
        "answer_templates": ["A good summary should explain the central idea of {topic}, avoid unnecessary detail, and keep the language concise."],
    },
    "classification": {
        "subject": "General", "category": "Classification", "subcategory": "topic_classification",
        "templates": [
            "Classify this request as informational, transactional, or creative: Please explain {topic} to {audience}.",
            "Classify this message as urgent, normal, or low priority: I need help with {service} before tomorrow morning.",
            "Classify this text as positive, neutral, or negative: The {product} works well, but the setup took longer than expected.",
            "Classify this question by domain: How does {science_concept} affect the environment?",
        ],
        "answer_templates": ["A good answer should choose the most appropriate label and give a short reason for the classification."],
    },
    "sentiment_classification": {
        "subject": "General", "category": "Classification", "subcategory": "sentiment",
        "templates": [
            "Classify the sentiment of this review: The {product} is useful, but the battery life could be better.",
            "Classify the sentiment of this message: I am very happy with the support I received for {service}.",
            "Classify the sentiment of this review: The delivery was late and the packaging was damaged.",
            "Classify the sentiment of this comment: The new update is acceptable, although it still needs improvement.",
        ],
        "answer_templates": ["A good answer should label the sentiment as positive, neutral, mixed, or negative and briefly justify the label."],
    },
    "information_extraction": {
        "subject": "General", "category": "Information Extraction", "subcategory": "key_fields",
        "templates": [
            "Extract the date, location, and main topic from this sentence: The workshop on {topic} will be held in {city} on Monday.",
            "Extract the product, issue, and requested action from this sentence: My {product} stopped working and I need a replacement.",
            "Extract the service name and user problem from this sentence: I cannot log into the {service} after changing my password.",
            "Extract the audience and topic from this sentence: This guide explains {topic} for {audience}.",
        ],
        "answer_templates": ["A good answer should list the requested fields clearly and copy only the relevant information from the sentence."],
    },
    "named_entity_extraction": {
        "subject": "General", "category": "Information Extraction", "subcategory": "named_entities",
        "templates": [
            "Extract all named entities from this sentence: The conference in {city} included speakers from {place} and France.",
            "Identify the person, organization, and location in this sentence: A researcher from the university visited {city} for a climate study.",
            "Extract places and organizations from this sentence: The public library in {city} hosted a workshop about {topic}.",
            "Find the location and event from this sentence: A technology exhibition was organized in {city} last week.",
        ],
        "answer_templates": ["A good answer should separate entities by type, such as person, organization, location, and event."],
    },
    "structured_json": {
        "subject": "General", "category": "Structured Output", "subcategory": "json_conversion",
        "templates": [
            "Convert this request into JSON with keys intent, topic, and audience: Explain {topic} to {audience}.",
            "Convert this support message into JSON with keys product, issue, and urgency: My {product} is not working before an important meeting.",
            "Create JSON with keys service, problem, and requested_action for this message: I cannot access my {service} account.",
            "Convert this sentence into JSON with keys city, event, and topic: A workshop about {topic} was held in {city}.",
        ],
        "answer_templates": ["A good answer should return valid JSON only, using the requested keys and concise values."],
    },
    "table_conversion": {
        "subject": "General", "category": "Structured Output", "subcategory": "table_conversion",
        "templates": [
            "Convert this information into a two-column table: Product is {product}; issue is delayed delivery; urgency is medium.",
            "Create a table with topic, audience, and tone for this request: Explain {topic} to {audience} in a {tone} tone.",
            "Convert this sentence into a table: The workshop topic is {topic}, the city is {city}, and the audience is {audience}.",
            "Make a simple table from this support case: service={service}, problem=login failure, action=reset password.",
        ],
        "answer_templates": ["A good answer should create a clear table with the requested fields and no extra explanation."],
    },
    "code_generation": {
        "subject": "Programming", "category": "Code Generation", "subcategory": "python_basics",
        "templates": [
            "Write a Python function that takes a list of numbers and returns the largest value.",
            "Write a Python function that counts how many words are in a sentence.",
            "Write a Python function that checks whether a number is even.",
            "Write a Python function that removes duplicate items from a list while keeping the original order.",
        ],
        "answer_templates": ["A good answer should provide a short Python function, include clear variable names, and show one simple example."],
    },
    "code_explanation": {
        "subject": "Programming", "category": "Code Explanation", "subcategory": "python_basics",
        "templates": [
            "Explain what this Python code does: for item in items: print(item)",
            "Explain what this Python expression means: len(text.split())",
            "Explain what a return statement does in a Python function.",
            "Explain why indentation matters in Python code.",
        ],
        "answer_templates": ["A good answer should explain the code behavior in plain language and mention the main programming concept involved."],
    },
    "debugging_help": {
        "subject": "Programming", "category": "Debugging", "subcategory": "common_errors",
        "templates": [
            "A Python program shows a TypeError when adding a string and an integer. What is the likely problem?",
            "A loop does not stop running. What are two possible causes?",
            "A file path works on one computer but not another. What should be checked?",
            "A function returns None unexpectedly. What is one likely reason?",
        ],
        "answer_templates": ["A good answer should identify the likely bug, explain why it happens, and suggest a safe fix."],
    },
    "professional_writing": {
        "subject": "Writing", "category": "Professional Writing", "subcategory": "workplace_message",
        "templates": [
            "Write a {tone} message asking a colleague to review a document about {topic}.",
            "Write a short professional note explaining a delay in {service}.",
            "Rewrite this message in a more professional tone: I need this task finished soon.",
            "Write a concise update for a team working on {topic}.",
        ],
        "answer_templates": ["A good answer should be professional, concise, respectful, and directly address the requested situation."],
    },
    "email_writing": {
        "subject": "Writing", "category": "Email Writing", "subcategory": "professional_email",
        "templates": [
            "Write a polite email asking to reschedule a meeting about {topic}.",
            "Write an email requesting more information about {service}.",
            "Write a short email thanking someone for help with {topic}.",
            "Write a follow-up email after a discussion about {topic}.",
        ],
        "answer_templates": ["A good answer should include a clear subject idea, polite greeting, concise body, and professional closing."],
    },
    "customer_support": {
        "subject": "Business", "category": "Customer Support", "subcategory": "support_reply",
        "templates": [
            "Write a customer support reply for a user whose {product} arrived damaged.",
            "Write a support response explaining how to reset a password for {service}.",
            "Write a polite response to a customer asking for a refund for {product}.",
            "Write a support message apologizing for a delay in {service}.",
        ],
        "answer_templates": ["A good answer should acknowledge the issue, apologize if needed, explain the next step, and use a helpful tone."],
    },
    "plain_language_legal": {
        "subject": "Legal", "category": "Plain Language Explanation", "subcategory": "legal_general_info",
        "templates": [
            "Explain in plain language what a contract is.",
            "Explain in plain language why reading terms and conditions matters.",
            "Explain in plain language what privacy consent means.",
            "Explain in plain language what a refund policy usually describes.",
        ],
        "answer_templates": ["A good answer should give general information in plain language and avoid giving personal legal advice."],
    },
    "plain_language_medical": {
        "subject": "Health", "category": "Plain Language Explanation", "subcategory": "health_general_info",
        "templates": [
            "Explain in plain language why sleep is important for health.",
            "Explain in plain language what hydration means.",
            "Explain in plain language why regular exercise can be helpful.",
            "Explain in plain language what a vaccine is generally used for.",
        ],
        "answer_templates": ["A good answer should provide general health information in simple language and avoid personal medical diagnosis."],
    },
    "plain_language_finance": {
        "subject": "Finance", "category": "Plain Language Explanation", "subcategory": "finance_general_info",
        "templates": [
            "Explain in plain language what a budget is.",
            "Explain in plain language why saving money regularly can help.",
            "Explain in plain language what interest means in a bank account.",
            "Explain in plain language what an invoice is used for.",
        ],
        "answer_templates": ["A good answer should provide general financial information in simple language and avoid personal financial advice."],
    },
    "education_tutoring": {
        "subject": "Education", "category": "Tutoring", "subcategory": "student_explanation",
        "templates": [
            "Explain {science_concept} to a high-school student using a simple example.",
            "Create a short study tip for learning about {topic}.",
            "Explain how a student can prepare for a quiz about {history_event}.",
            "Teach the basic idea of average speed using a simple example.",
        ],
        "answer_templates": ["A good answer should explain the idea step by step, use simple wording, and include a short example."],
    },
    "creative_writing": {
        "subject": "Writing", "category": "Creative Writing", "subcategory": "short_creative_text",
        "templates": [
            "Write a short story opening about {topic} in a hopeful tone.",
            "Write a creative paragraph about a traveler arriving in {city}.",
            "Write a short scene where a student learns about {science_concept}.",
            "Write a simple poem about {topic} for beginners.",
        ],
        "answer_templates": ["A good answer should be creative, coherent, and match the requested tone and topic."],
    },
    "safety_explanation": {
        "subject": "Safety", "category": "Safety Explanation", "subcategory": "digital_and_general_safety",
        "templates": [
            "Explain why sharing passwords with strangers is unsafe.",
            "Explain why checking a link before clicking it is important.",
            "Explain how to respond safely to a suspicious message about {service}.",
            "Explain why backing up important files is useful.",
        ],
        "answer_templates": ["A good answer should explain the risk clearly and suggest a safe, practical alternative."],
    },
    "comparison_question": {
        "subject": "General", "category": "Comparison", "subcategory": "compare_and_contrast",
        "templates": [
            "Compare {topic} and traditional classroom learning in two key ways.",
            "Compare public transport and private cars for daily travel.",
            "Compare renewable energy and non-renewable energy in simple terms.",
            "Compare online shopping and in-store shopping for buying a {product}.",
        ],
        "answer_templates": ["A good answer should mention at least two similarities or differences and present them clearly."],
    },
    "planning_question": {
        "subject": "Planning", "category": "Planning", "subcategory": "simple_plan",
        "templates": [
            "Create a simple three-step plan for learning about {topic}.",
            "Create a one-day plan for visiting {city} on a limited budget.",
            "Create a short plan for preparing a presentation about {topic}.",
            "Create a checklist for setting up a new {product}.",
        ],
        "answer_templates": ["A good answer should provide ordered steps, keep the plan realistic, and avoid unnecessary detail."],
    },
    "instruction_following": {
        "subject": "General", "category": "Instruction Following", "subcategory": "format_constraints",
        "templates": [
            "Answer in exactly three bullet points: Why is {topic} important?",
            "Explain {science_concept} in one sentence using simple words.",
            "List two advantages and two disadvantages of {service}.",
            "Give a yes-or-no answer first, then explain whether {topic} is useful for {audience}.",
        ],
        "answer_templates": ["A good answer should follow the requested format exactly and answer the question directly."],
    },
    "data_analysis_question": {
        "subject": "Data Analysis", "category": "Data Analysis", "subcategory": "basic_interpretation",
        "templates": [
            "A survey shows that 60 out of 100 students prefer online learning. What does this result mean?",
            "A store sold 40 units on Monday and 60 units on Tuesday. What changed between the two days?",
            "A chart shows that usage of {service} increased each month. What general trend does this suggest?",
            "A class average rose from 70 to 78. How much did it increase?",
        ],
        "answer_templates": ["A good answer should identify the key numbers, describe the trend or difference, and state the interpretation clearly."],
    },
    "cultural_localization": {
        "subject": "Localization", "category": "Cultural Localization", "subcategory": "local_adaptation",
        "templates": [
            "How would you adapt a message about {topic} for {audience} in a respectful way?",
            "What should be considered when explaining {service} to people in {place}?",
            "Why is cultural context important when writing about {topic}?",
            "How can a public message about {topic} be made clearer for a local community?",
        ],
        "answer_templates": ["A good answer should mention audience, language clarity, cultural context, and respectful wording."],
    },
    "translation_instruction": {
        "subject": "Language", "category": "Translation Instruction", "subcategory": "translation_request",
        "templates": [
            "Translate this short message from {language_a} to {language_b}: Thank you for your help today.",
            "Translate this sentence into {language_b}: The meeting will start at ten in the morning.",
            "Translate this polite request into {language_b}: Could you please send the document again?",
            "Translate this simple instruction into {language_b}: Please save the file before closing the program.",
        ],
        "answer_templates": ["A good answer should provide an accurate translation and preserve the tone of the original sentence."],
    },
    "fact_checking": {
        "subject": "General", "category": "Fact Checking", "subcategory": "claim_review",
        "templates": [
            "Check whether this claim needs verification: {claim}.",
            "Identify what evidence would be needed to verify this claim: {claim}.",
            "Explain why this statement should be checked before sharing: {claim}.",
            "List two questions that would help fact-check this claim: {claim}.",
        ],
        "answer_templates": ["A good answer should explain what part of the claim needs evidence and suggest a reliable way to verify it."],
    },
    "argument_analysis": {
        "subject": "Reasoning", "category": "Argument Analysis", "subcategory": "claim_reason_evidence",
        "templates": [
            "Analyze the argument that {claim}. What is the main claim and what evidence is missing?",
            "Identify the claim, reason, and possible weakness in this argument: {claim}.",
            "Explain whether this argument is strong or weak: {claim}.",
            "Give one counterpoint to the argument that {claim}.",
        ],
        "answer_templates": ["A good answer should identify the claim, discuss the evidence, and mention one limitation or counterpoint."],
    },
    "decision_making": {
        "subject": "Reasoning", "category": "Decision Making", "subcategory": "option_selection",
        "templates": [
            "Decide between {option_a} and {option_b} for {audience}. What factors matter most?",
            "Compare {option_a} and {option_b}, then recommend one for a small project.",
            "What questions should someone ask before choosing between {option_a} and {option_b}?",
            "Recommend whether {option_a} or {option_b} is better for beginners, and explain why.",
        ],
        "answer_templates": ["A good answer should compare the options, state the recommendation, and explain the main trade-off."],
    },
    "project_management": {
        "subject": "Work", "category": "Project Management", "subcategory": "task_planning",
        "templates": [
            "Create a simple project plan for completing work on {topic} in one week.",
            "List the main risks in a project about {topic} and how to reduce them.",
            "Prepare a task breakdown for a team working on {meeting_topic}.",
            "Suggest milestones for a small project about {topic}.",
        ],
        "answer_templates": ["A good answer should organize the project into clear tasks, milestones, risks, and next steps."],
    },
    "meeting_notes": {
        "subject": "Work", "category": "Meeting Notes", "subcategory": "summary_and_actions",
        "templates": [
            "Create meeting notes for a discussion about {meeting_topic}, including decisions and action items.",
            "Summarize a meeting about {topic} into agenda, key points, and next steps.",
            "Prepare action items after a team meeting about {meeting_topic}.",
            "Organize these meeting points into clear notes: timeline, blockers, owner, and next step for {topic}.",
        ],
        "answer_templates": ["A good answer should include concise notes, decisions, owners, and action items."],
    },
    "policy_explanation": {
        "subject": "Policy", "category": "Policy Explanation", "subcategory": "plain_policy",
        "templates": [
            "Explain a simple workplace policy about {policy_topic} in plain language.",
            "Describe what employees should remember about a policy on {policy_topic}.",
            "Rewrite a policy about {policy_topic} so it is easier for {audience} to understand.",
            "List the main do's and don'ts for a policy about {policy_topic}.",
        ],
        "answer_templates": ["A good answer should explain the policy clearly, avoid legal jargon, and give practical guidance."],
    },
    "research_question": {
        "subject": "Research", "category": "Research Design", "subcategory": "question_design",
        "templates": [
            "Suggest a research question about {topic} for a small student project.",
            "Turn this broad topic into a focused research question: {topic}.",
            "Create two measurable research questions about {topic}.",
            "Explain how to make a research question about {topic} more specific.",
        ],
        "answer_templates": ["A good answer should produce a focused, measurable research question and explain why it is specific."],
    },
    "data_cleaning_instruction": {
        "subject": "Data Analysis", "category": "Data Cleaning", "subcategory": "cleaning_steps",
        "templates": [
            "Describe how to handle {dataset_issue} in a dataset before analysis.",
            "List steps to clean a dataset with {dataset_issue}.",
            "Explain why {dataset_issue} can affect data analysis results.",
            "Prepare a short checklist for finding and fixing {dataset_issue}.",
        ],
        "answer_templates": ["A good answer should explain the data issue, describe cleaning steps, and mention why the fix matters."],
    },
    "algorithmic_reasoning": {
        "subject": "Computer Science", "category": "Algorithmic Reasoning", "subcategory": "stepwise_logic",
        "templates": [
            "Explain how to find the largest number in a list step by step.",
            "Describe an algorithm for counting repeated words in a sentence.",
            "Explain how binary search works using a simple example.",
            "Describe how to sort a small list of numbers in increasing order.",
        ],
        "answer_templates": ["A good answer should describe the algorithm step by step and include a simple example or complexity intuition."],
    },
    "multilingual_localization_advice": {
        "subject": "Localization", "category": "Localization Advice", "subcategory": "translation_and_culture",
        "templates": [
            "Suggest how to localize a message about {topic} for speakers of {language_b}.",
            "Explain what to consider when adapting a customer message for {place}.",
            "Describe how tone and cultural context can affect translation of a message about {service}.",
            "Recommend how to make a product message about {product} clearer for a multilingual audience.",
        ],
        "answer_templates": ["A good answer should mention language clarity, cultural fit, tone, and avoiding literal translation when needed."],
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def clean_text(text: Any) -> str:
    text = str(text).strip()
    text = text.replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"\s+", " ", text)
    return text


def is_bad_prompt(text: str) -> bool:
    t = clean_text(text)
    lower = t.lower()
    if len(t) < 12:
        return True
    bad_patterns = [
        "...", "blah", "do do", "door do", "my name is", "lorem ipsum",
        "{", "}", "[placeholder", "insert", "undefined", "none", "nan",
    ]
    if any(p in lower for p in bad_patterns):
        return True
    if re.search(r"\b[a-zA-Z]+\s*/\s*[a-zA-Z]+\b", t):
        return True
    if t.count("?") > 2:
        return True
    # Ensure it is a real instruction or question.
    starters = (
        "what", "why", "how", "if", "a ", "an ", "write", "rewrite", "summarize",
        "classify", "extract", "identify", "convert", "create", "explain", "list",
        "compare", "give", "translate", "answer", "make",
        "check", "analyze", "suggest", "draft", "prepare", "review",
        "recommend", "describe", "organize", "decide", "evaluate",
        # Allow safe prompt-variation prefixes used by the dataset generator.
        "in ", "for ", "using ", "with ",
    )
    if not lower.startswith(starters):
        return True
    return False


def safe_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_")[:120]


def ensure_output_dir(path: str, clean: bool = False) -> Path:
    out = Path(path)
    if clean and out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "charts").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


def disk_report(label: str = "") -> None:
    try:
        usage = shutil.disk_usage("/kaggle/working")
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        print(f"Disk {label}: used={used_gb:.2f}GB free={free_gb:.2f}GB total={total_gb:.2f}GB")
    except Exception as e:
        print("Disk report failed:", repr(e))


def validate_paths() -> None:
    print("\nChecking model paths...")
    keys = ["qwen_model_path", "nllb_model_path", "labse_encoder_path", "labse_preprocessor_path"]
    missing = []
    for key in keys:
        p = CONFIG[key]
        exists = os.path.exists(p)
        print(f"{key}: {p} | exists={exists}")
        if not exists:
            missing.append((key, p))
    if missing:
        raise FileNotFoundError(f"Missing paths: {missing}")
    print("All configured paths exist.\n")


def print_environment() -> None:
    print("Environment check:")
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"gpu memory: {total:.2f} GB")
    except Exception as e:
        print("torch check failed:", repr(e))
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers check failed:", repr(e))


def fill_template(template: str, values: Dict[str, str]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return clean_text(out)


def required_placeholders(template: str) -> List[str]:
    return re.findall(r"\{([a-zA-Z0-9_]+)\}", template)


def value_for_slot(slot: str, idx: int) -> str:
    if slot in POOLS:
        pool = POOLS[slot]
    else:
        pool = [slot.replace("_", " ")]
    return pool[idx % len(pool)]


# ============================================================
# PROMPT DATASET CREATION
# ============================================================

def generate_prompt_rows() -> pd.DataFrame:
    rows = []
    seen_prompts = set()

    for category_key in CONFIG["categories_to_run"]:
        if category_key not in CATEGORY_SPECS:
            raise ValueError(f"Missing CATEGORY_SPECS for: {category_key}")

        spec = CATEGORY_SPECS[category_key]
        templates = spec["templates"]
        answer_templates = spec.get("answer_templates", ["A good answer should respond clearly and completely to the prompt."])
        needed = int(CONFIG["num_prompts_per_category"])
        made = 0
        attempt = 0
        max_attempts = needed * 20

        while made < needed and attempt < max_attempts:
            template = templates[attempt % len(templates)]
            placeholders = required_placeholders(template)

            # Use a slower-changing cycle for placeholder values and a separate
            # cycle for prompt wording variants. This lets every category create
            # 100+ complete, unique prompts even if it starts with only 4 base templates.
            base_cycle = attempt // max(1, len(templates))
            values = {ph: value_for_slot(ph, base_cycle) for ph in placeholders}

            base_prompt = fill_template(template, values)
            prefix = PROMPT_VARIANT_PREFIXES[(attempt // max(1, len(templates))) % len(PROMPT_VARIANT_PREFIXES)]
            suffix = PROMPT_VARIANT_SUFFIXES[(attempt // max(1, len(templates) * len(PROMPT_VARIANT_PREFIXES))) % len(PROMPT_VARIANT_SUFFIXES)]
            prompt = clean_text(prefix + base_prompt + suffix)

            answer_template = answer_templates[attempt % len(answer_templates)]
            answer_placeholders = required_placeholders(answer_template)
            for ph in answer_placeholders:
                if ph not in values:
                    values[ph] = value_for_slot(ph, attempt + made)
            reference_output = fill_template(answer_template, values)

            # Final cleanup and validation.
            prompt = clean_text(prompt)
            reference_output = clean_text(reference_output)

            if not is_bad_prompt(prompt) and prompt.lower() not in seen_prompts:
                made += 1
                seen_prompts.add(prompt.lower())
                prompt_id = f"{category_key.upper()}_{made:04d}"
                rows.append({
                    "prompt_id": prompt_id,
                    "category_key": category_key,
                    "subject": spec["subject"],
                    "category": spec["category"],
                    "subcategory": spec["subcategory"],
                    "source_language": "English",
                    "english_prompt": prompt,
                    "english_reference_output": reference_output,
                    "output_generation_mode": "deterministic_reference_output",
                })

            attempt += 1

        if made < needed:
            raise RuntimeError(f"Could only create {made}/{needed} prompts for category {category_key}")

    df = pd.DataFrame(rows)
    bad = df[df["english_prompt"].apply(is_bad_prompt)]
    if len(bad) > 0:
        bad_path = Path(CONFIG["output_dir"]) / "bad_prompt_debug.csv"
        bad.to_csv(bad_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Bad prompts detected. Saved: {bad_path}")
    return df


# ============================================================
# NLLB TRANSLATION
# ============================================================

def load_nllb():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_path = CONFIG["nllb_model_path"]
    print("Loading NLLB translation model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, dtype=dtype)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("NLLB loaded on:", device)
    return tokenizer, model, device


def translate_nllb(text: str, src_lang: str, tgt_lang: str, tokenizer, model, device: str) -> Tuple[str, float]:
    import torch
    text = clean_text(text)
    if not text:
        return "", 0.0
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Invalid NLLB target language code: {tgt_lang}")
    start = time.time()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=CONFIG["translation_max_new_tokens"],
            num_beams=CONFIG["translation_num_beams"],
            early_stopping=True,
        )
    elapsed_ms = round((time.time() - start) * 1000, 2)
    output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return clean_text(output), elapsed_ms


def read_completed_pairs(path: Path, key_cols: List[str]) -> set:
    if not path.exists() or not CONFIG.get("resume_translations", True):
        return set()
    try:
        df = pd.read_csv(path, usecols=key_cols)
        return set(tuple(row) for row in df[key_cols].astype(str).values.tolist())
    except Exception:
        return set()


def append_or_write_csv(path: Path, df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    if path.exists():
        df.to_csv(path, index=False, mode="a", header=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def run_prompt_translation_pipeline(prompts_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "02_prompt_translations_backtranslations.csv"
    key_cols = ["prompt_id", "target_language"]
    completed = read_completed_pairs(path, key_cols)

    tokenizer, model, device = load_nllb()
    rows_buffer = []
    total_jobs = len(prompts_df) * len(CONFIG["target_languages"])
    job_counter = 0
    skipped = 0

    print(f"Prompt translation jobs: {total_jobs}, already completed: {len(completed)}")

    for _, row in prompts_df.iterrows():
        english_prompt = row["english_prompt"]
        for language in CONFIG["target_languages"]:
            job_counter += 1
            pair_key = (str(row["prompt_id"]), str(language))
            if pair_key in completed:
                skipped += 1
                continue

            if job_counter == 1 or job_counter % CONFIG["progress_print_every"] == 0:
                print(f"Prompt translation progress {job_counter}/{total_jobs} | skipped={skipped}")

            src_code = NLLB_LANG_CODES["English"]
            tgt_code = NLLB_LANG_CODES[language]
            try:
                translated, forward_ms = translate_nllb(english_prompt, src_code, tgt_code, tokenizer, model, device)
                back_translation, backward_ms = translate_nllb(translated, tgt_code, src_code, tokenizer, model, device)
                status, error = "ok", ""
            except Exception as e:
                translated, back_translation = "", ""
                forward_ms, backward_ms = None, None
                status, error = "error", repr(e)

            rows_buffer.append({
                "prompt_id": row["prompt_id"],
                "category_key": row["category_key"],
                "subject": row["subject"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "source_language": "English",
                "target_language": language,
                "source_lang_code": src_code,
                "target_lang_code": tgt_code,
                "english_prompt": english_prompt,
                "translated_prompt": translated,
                "back_translation": back_translation,
                "forward_translation_ms": forward_ms,
                "back_translation_ms": backward_ms,
                "translation_status": status,
                "translation_error": error,
            })

            if len(rows_buffer) >= CONFIG["translation_checkpoint_every"]:
                append_or_write_csv(path, pd.DataFrame(rows_buffer))
                print(f"Checkpoint saved: {path} (+{len(rows_buffer)} rows)")
                rows_buffer = []
                gc.collect()

    append_or_write_csv(path, pd.DataFrame(rows_buffer))
    print(f"Prompt translations saved: {path}")
    return pd.read_csv(path)


def run_output_translation_pipeline(prompts_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "02B_output_translations.csv"

    if not CONFIG.get("include_output_side_analysis", True) or not CONFIG.get("translate_reference_outputs", True):
        rows = []
        for _, row in prompts_df.iterrows():
            for language in CONFIG["target_languages"]:
                rows.append({
                    "prompt_id": row["prompt_id"],
                    "target_language": language,
                    "english_reference_output": row["english_reference_output"],
                    "translated_reference_output": "",
                    "reference_output_translation_ms": None,
                    "output_translation_status": "disabled",
                    "output_translation_error": "",
                })
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return pd.DataFrame(rows)

    key_cols = ["prompt_id", "target_language"]
    completed = read_completed_pairs(path, key_cols)

    tokenizer, model, device = load_nllb()
    rows_buffer = []
    total_jobs = len(prompts_df) * len(CONFIG["target_languages"])
    job_counter = 0
    skipped = 0

    print(f"Output translation jobs: {total_jobs}, already completed: {len(completed)}")

    for _, row in prompts_df.iterrows():
        english_output = row["english_reference_output"]
        for language in CONFIG["target_languages"]:
            job_counter += 1
            pair_key = (str(row["prompt_id"]), str(language))
            if pair_key in completed:
                skipped += 1
                continue

            if job_counter == 1 or job_counter % CONFIG["progress_print_every"] == 0:
                print(f"Output translation progress {job_counter}/{total_jobs} | skipped={skipped}")

            src_code = NLLB_LANG_CODES["English"]
            tgt_code = NLLB_LANG_CODES[language]
            try:
                translated_output, ms = translate_nllb(english_output, src_code, tgt_code, tokenizer, model, device)
                status, error = "ok", ""
            except Exception as e:
                translated_output, ms = "", None
                status, error = "error", repr(e)

            rows_buffer.append({
                "prompt_id": row["prompt_id"],
                "target_language": language,
                "english_reference_output": english_output,
                "translated_reference_output": translated_output,
                "reference_output_translation_ms": ms,
                "output_translation_status": status,
                "output_translation_error": error,
            })

            if len(rows_buffer) >= CONFIG["translation_checkpoint_every"]:
                append_or_write_csv(path, pd.DataFrame(rows_buffer))
                print(f"Output checkpoint saved: {path} (+{len(rows_buffer)} rows)")
                rows_buffer = []
                gc.collect()

    append_or_write_csv(path, pd.DataFrame(rows_buffer))
    print(f"Output translations saved: {path}")
    return pd.read_csv(path)


# ============================================================
# LaBSE SIMILARITY
# ============================================================

class LabseScorer:
    def __init__(self, encoder_path: str, preprocessor_path: str, batch_size: int = 128):
        self.encoder_path = encoder_path
        self.preprocessor_path = preprocessor_path
        self.batch_size = batch_size
        self.method = "labse_tfhub_preprocessor_encoder"
        self.encoder = None
        self.preprocessor = None
        self._load()

    def _load(self) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub
        import tensorflow_text as text  # noqa: F401
        try:
            tf.config.set_visible_devices([], "GPU")
            print("TensorFlow GPU disabled for LaBSE; using CPU for semantic similarity.")
        except Exception:
            pass
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(self.preprocessor_path)
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(self.encoder_path)
        self.preprocessor = hub.KerasLayer(self.preprocessor_path)
        self.encoder = hub.KerasLayer(self.encoder_path)
        rel = self.similarity_pair("What is the main function of the human heart?", "What is the primary function of the human heart?")
        unrel = self.similarity_pair("What is the main function of the human heart?", "The weather is sunny today.")
        print(f"LaBSE test related={rel:.4f} unrelated={unrel:.4f}. Semantic similarity active.")

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def encode(self, sentences: List[str]) -> np.ndarray:
        import tensorflow as tf
        all_embeddings = []
        for start in range(0, len(sentences), self.batch_size):
            batch = [clean_text(s) for s in sentences[start:start + self.batch_size]]
            embeds = self.encoder(self.preprocessor(tf.constant(batch)))["default"].numpy().astype("float32")
            all_embeddings.append(embeds)
        return self._normalize(np.vstack(all_embeddings))

    def similarity_pair(self, a: str, b: str) -> float:
        emb = self.encode([a, b])
        return float(np.dot(emb[0], emb[1]))


def add_similarity_scores(trans_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    scored_path = out_dir / "03_similarity_scored.csv"
    print("Loading LaBSE for prompt back-translation similarity...")
    scorer = LabseScorer(CONFIG["labse_encoder_path"], CONFIG["labse_preprocessor_path"], CONFIG["labse_batch_size"])

    pairs = trans_df[["english_prompt", "back_translation"]].copy()
    sentences = []
    for _, row in pairs.iterrows():
        sentences.append(row["english_prompt"])
        sentences.append(row["back_translation"])

    print(f"Encoding {len(sentences)} sentences with LaBSE...")
    emb = scorer.encode(sentences)
    scores = []
    for i in range(0, len(emb), 2):
        score = float(np.dot(emb[i], emb[i + 1]))
        scores.append(round(score, 4))

    scored = trans_df.copy()
    scored["similarity_score"] = scores
    scored["similarity_method"] = scorer.method
    scored["similarity_threshold"] = CONFIG["similarity_threshold"]
    scored["auto_quality_status"] = np.where(scored["similarity_score"] >= CONFIG["similarity_threshold"], "pass", "review")
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    print(f"Similarity scored saved: {scored_path}")
    return scored


# ============================================================
# TOKEN COUNTING AND MASTER BUILD
# ============================================================

def load_tokenizers() -> Dict[str, object]:
    from transformers import AutoTokenizer
    tokenizers = {}
    if "qwen3_14b_base" in CONFIG["tokenizers_to_use"]:
        tokenizers["qwen3_14b_base"] = AutoTokenizer.from_pretrained(CONFIG["qwen_model_path"], local_files_only=True, trust_remote_code=True)
    if "nllb_200_distilled_1_3b" in CONFIG["tokenizers_to_use"]:
        tokenizers["nllb_200_distilled_1_3b"] = AutoTokenizer.from_pretrained(CONFIG["nllb_model_path"], local_files_only=True)
    print("Loaded tokenizers:", list(tokenizers.keys()))
    return tokenizers


def count_tokens(tokenizer, text: Any) -> int:
    text = clean_text(text)
    if not text:
        return 0
    try:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    except TypeError:
        return int(len(tokenizer.encode(text)))


def build_dataset_main(scored_df: pd.DataFrame, output_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    merged = scored_df.merge(
        output_df,
        on=["prompt_id", "target_language"],
        how="left",
    )
    # Avoid duplicate english_reference_output columns if they appear.
    if "english_reference_output_y" in merged.columns:
        merged["english_reference_output"] = merged["english_reference_output_y"].fillna(merged.get("english_reference_output_x", ""))
        merged = merged.drop(columns=[c for c in ["english_reference_output_x", "english_reference_output_y"] if c in merged.columns])
    path = out_dir / "00_DATASET_MAIN_NO_TOKENIZER_DUPLICATES.csv"
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Dataset main saved: {path} rows={len(merged)}")
    return merged


def build_master(dataset_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    tokenizers = load_tokenizers()
    rows = []
    total = len(dataset_df)

    for idx, row in dataset_df.iterrows():
        if idx == 0 or (idx + 1) % 1000 == 0:
            print(f"Token counting progress {idx + 1}/{total}")

        english_prompt = row.get("english_prompt", "")
        translated_prompt = row.get("translated_prompt", "")
        english_output = row.get("english_reference_output", "")
        translated_output = row.get("translated_reference_output", "")

        for tok_name, tok in tokenizers.items():
            input_en = count_tokens(tok, english_prompt)
            input_tgt = count_tokens(tok, translated_prompt)

            if CONFIG.get("include_output_side_analysis", True):
                output_en = count_tokens(tok, english_output)
                output_tgt = count_tokens(tok, translated_output)
            else:
                output_en = 0
                output_tgt = 0

            total_en = input_en + output_en
            total_tgt = input_tgt + output_tgt

            new_row = row.to_dict()
            new_row.update({
                "tokenizer_name": tok_name,
                "input_english_token_count": input_en,
                "input_translated_token_count": input_tgt,
                "output_english_token_count": output_en,
                "output_translated_token_count": output_tgt,
                "total_english_token_count": total_en,
                "total_translated_token_count": total_tgt,
                "input_ter_vs_english": round(input_tgt / input_en, 4) if input_en > 0 else None,
                "output_ter_vs_english": round(output_tgt / output_en, 4) if output_en > 0 else None,
                "total_ter_vs_english": round(total_tgt / total_en, 4) if total_en > 0 else None,
                "target_input_token_share": round(input_tgt / total_tgt, 4) if total_tgt > 0 else None,
                "target_output_token_share": round(output_tgt / total_tgt, 4) if total_tgt > 0 else None,
            })
            # Backward compatibility aliases.
            new_row["english_token_count"] = input_en
            new_row["translated_token_count"] = input_tgt
            new_row["ter_vs_english"] = new_row["input_ter_vs_english"]
            rows.append(new_row)

    master = pd.DataFrame(rows)
    path = out_dir / "00_MASTER_FOR_REVIEW.csv"
    master.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Master saved: {path} rows={len(master)}")
    return master


# ============================================================
# SUMMARIES AND REVIEW FILES
# ============================================================

def summarize(master_df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Overall by language/tokenizer.
    overall = master_df.groupby(["tokenizer_name", "target_language"], dropna=False).agg(
        samples=("prompt_id", "nunique"),
        mean_input_english_tokens=("input_english_token_count", "mean"),
        mean_input_translated_tokens=("input_translated_token_count", "mean"),
        mean_output_english_tokens=("output_english_token_count", "mean"),
        mean_output_translated_tokens=("output_translated_token_count", "mean"),
        mean_total_english_tokens=("total_english_token_count", "mean"),
        mean_total_translated_tokens=("total_translated_token_count", "mean"),
        mean_input_ter_vs_english=("input_ter_vs_english", "mean"),
        mean_output_ter_vs_english=("output_ter_vs_english", "mean"),
        mean_total_ter_vs_english=("total_ter_vs_english", "mean"),
        mean_target_input_token_share=("target_input_token_share", "mean"),
        mean_target_output_token_share=("target_output_token_share", "mean"),
        mean_similarity_score=("similarity_score", "mean"),
        review_count=("auto_quality_status", lambda x: int((x == "review").sum())),
        pass_count=("auto_quality_status", lambda x: int((x == "pass").sum())),
        mean_forward_ms=("forward_translation_ms", "mean"),
        mean_back_ms=("back_translation_ms", "mean"),
    ).reset_index()

    cat = master_df.groupby(["tokenizer_name", "category_key", "target_language"], dropna=False).agg(
        samples=("prompt_id", "nunique"),
        mean_input_ter_vs_english=("input_ter_vs_english", "mean"),
        mean_output_ter_vs_english=("output_ter_vs_english", "mean"),
        mean_total_ter_vs_english=("total_ter_vs_english", "mean"),
        mean_similarity_score=("similarity_score", "mean"),
        review_count=("auto_quality_status", lambda x: int((x == "review").sum())),
    ).reset_index()

    for df in [overall, cat]:
        for c in df.columns:
            if c.startswith("mean_"):
                df[c] = df[c].round(4)

    overall = overall.sort_values(["tokenizer_name", "mean_total_ter_vs_english"])
    cat = cat.sort_values(["tokenizer_name", "category_key", "mean_total_ter_vs_english"])

    overall_path = out_dir / "05B_overall_language_summary.csv"
    cat_path = out_dir / "05_summary_by_language_category_tokenizer.csv"
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    cat.to_csv(cat_path, index=False, encoding="utf-8-sig")
    print(f"Summaries saved: {overall_path}, {cat_path}")

    return overall, cat


def save_review_files(dataset_df: pd.DataFrame, master_df: pd.DataFrame, out_dir: Path) -> None:
    review_cols = [
        "prompt_id", "category_key", "subject", "category", "subcategory", "target_language",
        "english_prompt", "translated_prompt", "back_translation", "english_reference_output",
        "translated_reference_output", "similarity_score", "similarity_method", "auto_quality_status",
        "translation_status", "translation_error", "output_translation_status", "output_translation_error",
    ]
    existing = [c for c in review_cols if c in dataset_df.columns]
    review = dataset_df[existing].drop_duplicates(["prompt_id", "target_language"]).copy()
    review["human_review_status"] = ""
    review["human_notes"] = ""
    review = review.sort_values(["auto_quality_status", "similarity_score", "target_language"], ascending=[True, True, True])

    review_path = out_dir / "06_human_review_sheet.csv"
    low_path = out_dir / "07_low_similarity_review_shortlist.csv"
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    review[review["auto_quality_status"] == "review"].to_csv(low_path, index=False, encoding="utf-8-sig")

    # Save prompt-quality issue file.
    prompt_issues = dataset_df[dataset_df["english_prompt"].apply(is_bad_prompt)].copy()
    prompt_issues.to_csv(out_dir / "09_prompt_quality_issues.csv", index=False, encoding="utf-8-sig")

    print(f"Review files saved: {review_path}, {low_path}")


# ============================================================
# CHARTS
# ============================================================

def save_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, path: Path, hue_col: Optional[str] = None, baseline: Optional[float] = None) -> None:
    if plt is None or len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    if hue_col is None:
        plot_df = df.sort_values(y_col)
        ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
    else:
        labels = sorted(df[x_col].astype(str).unique().tolist())
        hues = sorted(df[hue_col].astype(str).unique().tolist())
        width = 0.8 / max(1, len(hues))
        x = np.arange(len(labels))
        for i, hue in enumerate(hues):
            vals = []
            sub = df[df[hue_col].astype(str) == hue]
            for lab in labels:
                row = sub[sub[x_col].astype(str) == lab]
                vals.append(float(row[y_col].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - (len(hues)-1)/2) * width, vals, width, label=hue)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend(title=hue_col)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, tokenizer_name: str, value_col: str, title: str, path: Path) -> None:
    if plt is None:
        return
    sub = df[df["tokenizer_name"] == tokenizer_name].copy()
    if len(sub) == 0:
        return
    pivot = sub.pivot_table(index="category_key", columns="target_language", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_charts(overall: pd.DataFrame, category_summary: pd.DataFrame, out_dir: Path) -> None:
    if not CONFIG.get("save_charts", True):
        return
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_bar_chart(
            overall, "target_language", "mean_input_ter_vs_english",
            "Input TER by Language and Tokenizer", "Mean Input TER vs English",
            charts_dir / "input_ter_by_language_tokenizer.png", hue_col="tokenizer_name", baseline=1.0,
        )
        save_bar_chart(
            overall, "target_language", "mean_output_ter_vs_english",
            "Output TER by Language and Tokenizer", "Mean Output TER vs English",
            charts_dir / "output_ter_by_language_tokenizer.png", hue_col="tokenizer_name", baseline=1.0,
        )
        save_bar_chart(
            overall, "target_language", "mean_total_ter_vs_english",
            "Total TER by Language and Tokenizer", "Mean Total TER vs English",
            charts_dir / "total_ter_by_language_tokenizer.png", hue_col="tokenizer_name", baseline=1.0,
        )
        sim = overall.groupby("target_language", as_index=False).agg(mean_similarity_score=("mean_similarity_score", "mean"))
        save_bar_chart(
            sim, "target_language", "mean_similarity_score",
            "Mean Back-Translation Similarity by Language", "Mean LaBSE Similarity",
            charts_dir / "similarity_by_language.png", baseline=CONFIG["similarity_threshold"],
        )
        qwen = overall[overall["tokenizer_name"] == "qwen3_14b_base"].copy()
        save_bar_chart(
            qwen, "target_language", "mean_total_ter_vs_english",
            "Qwen3 Total Token Efficiency Ratio vs English", "Mean Total TER vs English",
            charts_dir / "qwen_total_ter_by_language.png", baseline=1.0,
        )
        save_heatmap(
            category_summary, "qwen3_14b_base", "mean_total_ter_vs_english",
            "Qwen3 Total TER Heatmap by Category and Language",
            charts_dir / "qwen_total_ter_heatmap.png",
        )
        save_heatmap(
            category_summary, "qwen3_14b_base", "mean_input_ter_vs_english",
            "Qwen3 Input TER Heatmap by Category and Language",
            charts_dir / "qwen_input_ter_heatmap.png",
        )
        print(f"Charts saved to: {charts_dir}")
    except Exception as e:
        print("Chart generation failed but pipeline will continue:", repr(e))


# ============================================================
# FINALIZATION
# ============================================================

def write_status(out_dir: Path, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "status": status,
        "run_preset": RUN_PRESET,
        "run_name": CONFIG["run_name"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": CONFIG,
    }
    if extra:
        payload.update(extra)
    with open(out_dir / "08_PIPELINE_STATUS.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def maybe_make_zip(out_dir: Path) -> None:
    if not CONFIG.get("make_zip_bundle", False):
        print("Zip bundle disabled for this run to save disk space.")
        return
    try:
        zip_path = shutil.make_archive(str(out_dir / "token_efficiency_outputs_bundle"), "zip", str(out_dir))
        print("Zip bundle saved:", zip_path)
    except Exception as e:
        print("Zip creation failed but outputs are saved:", repr(e))


def save_readme(out_dir: Path) -> None:
    readme = f"""CPM-Bench output summary
========================
Run preset: {RUN_PRESET}
Run name: {CONFIG['run_name']}
Categories: {len(CONFIG['categories_to_run'])}
Prompts per category: {CONFIG['num_prompts_per_category']}
Languages: {', '.join(CONFIG['target_languages'])}
Input-output analysis: {CONFIG.get('include_output_side_analysis', True)}

Key files:
- 00_DATASET_MAIN_NO_TOKENIZER_DUPLICATES.csv: one row per prompt-language pair.
- 00_MASTER_FOR_REVIEW.csv: one row per prompt-language-tokenizer pair, includes input/output/total token counts.
- 05B_overall_language_summary.csv: overall language-level summary.
- 05_summary_by_language_category_tokenizer.csv: category-level summary.
- 06_human_review_sheet.csv: manual review sheet.
- 07_low_similarity_review_shortlist.csv: low-similarity rows needing review.
- charts/: saved figures.

Important metrics:
- input_ter_vs_english = translated prompt tokens / English prompt tokens.
- output_ter_vs_english = translated output tokens / English output tokens.
- total_ter_vs_english = translated input+output tokens / English input+output tokens.
- target_input_token_share = share of total target tokens from the prompt.
- target_output_token_share = share of total target tokens from the output.
"""
    with open(out_dir / "README_OUTPUTS.txt", "w", encoding="utf-8") as f:
        f.write(readme)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    start_time = time.time()
    set_seed(42)

    out_dir = ensure_output_dir(CONFIG["output_dir"], clean=CONFIG.get("clean_output_dir_on_start", False))
    print("=" * 90)
    print("CPM-BENCH SCALE PIPELINE: INPUT + OUTPUT TOKEN EFFICIENCY")
    print("=" * 90)
    print("RUN_PRESET:", RUN_PRESET)
    print("Run name:", CONFIG["run_name"])
    print("Categories:", len(CONFIG["categories_to_run"]), CONFIG["categories_to_run"])
    print("Prompts/category:", CONFIG["num_prompts_per_category"])
    print("Languages:", CONFIG["target_languages"])
    print("Output-side analysis:", CONFIG.get("include_output_side_analysis", True))
    print("=" * 90)

    try:
        print_environment()
        disk_report("at start")
        validate_paths()
        with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, ensure_ascii=False, indent=2)

        # 1. Prompt dataset
        prompts_df = generate_prompt_rows()
        prompts_path = out_dir / "01_english_prompts.csv"
        prompts_df.to_csv(prompts_path, index=False, encoding="utf-8-sig")
        print(f"English prompts saved: {prompts_path} rows={len(prompts_df)}")

        # 2. Prompt translations + backtranslations
        prompt_trans_df = run_prompt_translation_pipeline(prompts_df, out_dir)
        err_count = int((prompt_trans_df["translation_status"] != "ok").sum()) if "translation_status" in prompt_trans_df.columns else -1
        print("Prompt translation errors:", err_count)

        # 3. Output/reference answer translations
        output_df = run_output_translation_pipeline(prompts_df, out_dir)
        if "output_translation_status" in output_df.columns:
            out_err = int((output_df["output_translation_status"] == "error").sum())
            print("Output translation errors:", out_err)

        # Release NLLB GPU memory before LaBSE/token counting.
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 4. Similarity scores
        scored_df = add_similarity_scores(prompt_trans_df, out_dir)

        # 5. Dataset main
        dataset_df = build_dataset_main(scored_df, output_df, out_dir)

        # 6. Master with token counts
        master_df = build_master(dataset_df, out_dir)

        # 7. Summaries/review/charts
        overall, category_summary = summarize(master_df, out_dir)
        save_review_files(dataset_df, master_df, out_dir)
        save_charts(overall, category_summary, out_dir)
        save_readme(out_dir)
        maybe_make_zip(out_dir)

        elapsed_min = round((time.time() - start_time) / 60, 2)
        write_status(out_dir, "success", {
            "elapsed_minutes": elapsed_min,
            "prompt_rows": int(len(prompts_df)),
            "dataset_main_rows": int(len(dataset_df)),
            "master_rows": int(len(master_df)),
            "translation_errors": err_count,
        })

        disk_report("at end")
        print("=" * 90)
        print("DONE SUCCESSFULLY")
        print("Elapsed minutes:", elapsed_min)
        print("Main dataset:", out_dir / "00_DATASET_MAIN_NO_TOKENIZER_DUPLICATES.csv")
        print("Master:", out_dir / "00_MASTER_FOR_REVIEW.csv")
        print("Overall summary:", out_dir / "05B_overall_language_summary.csv")
        print("Charts:", out_dir / "charts")
        print("=" * 90)

    except Exception as e:
        elapsed_min = round((time.time() - start_time) / 60, 2)
        write_status(out_dir, "failed", {"elapsed_minutes": elapsed_min, "error": repr(e)})
        print("PIPELINE FAILED:", repr(e))
        print("Status saved to:", out_dir / "08_PIPELINE_STATUS.json")
        raise


if __name__ == "__main__":
    main()
