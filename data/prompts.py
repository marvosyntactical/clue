"""Task configurations: dataset paths, prompt templates, label maps, and metrics.

Each task config contains everything needed to load, format, and evaluate one
CL task.  The TASK_ORDERS dict encodes the exact orderings from the paper
(Table 17).
"""

# ---------------------------------------------------------------------------
# Input builders — each takes a raw HF example dict and returns a string
# ---------------------------------------------------------------------------


def _text_only(key):
    def builder(ex):
        return ex[key]
    return builder


def _nli_builder(premise_key, hypothesis_key):
    def builder(ex):
        return f"Premise: {ex[premise_key]}\nHypothesis: {ex[hypothesis_key]}"
    return builder


def _boolq_builder(ex):
    return f"Passage: {ex['passage']}\nQuestion: {ex['question']}"


def _wic_builder(ex):
    return (
        f"Word: {ex['word']}\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}"
    )


def _copa_builder(ex):
    return (
        f"Premise: {ex['premise']}\n"
        f"Choice 1: {ex['choice1']}\n"
        f"Choice 2: {ex['choice2']}\n"
        f"Question: What is the {'cause' if ex['question'] == 'cause' else 'effect'}?"
    )


def _multirc_builder(ex):
    return (
        f"Passage: {ex['paragraph']}\n"
        f"Question: {ex['question']}\n"
        f"Answer: {ex['answer']}"
    )


def _yahoo_builder(ex):
    title = ex.get("question_title", "") or ""
    body = ex.get("question_content", "") or ""
    answer = ex.get("best_answer", "") or ""
    parts = [p for p in [title, body, answer] if p]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    # ---- CL Benchmark (5 tasks) ----
    "ag": {
        "hf_dataset": "ag_news",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _text_only("text"),
        "label_key": "label",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "instruction": (
            "Classify the following news article into one of the categories: "
            "World, Sports, Business, Sci/Tech."
        ),
        "metric": "accuracy",
    },
    "amazon": {
        # NOTE: The canonical 5-class Amazon Reviews from Zhang et al. (2015)
        # is not under a single HF name.  We point at yelp_review_full as a
        # structural stand-in (identical label cardinality / schema).  Replace
        # hf_dataset before running real experiments if you have a local copy.
        "hf_dataset": "yelp_review_full",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _text_only("text"),
        "label_key": "label",
        "label_names": ["1", "2", "3", "4", "5"],
        "instruction": (
            "Rate the sentiment of the following Amazon review from 1 "
            "(most negative) to 5 (most positive)."
        ),
        "metric": "accuracy",
    },
    "yelp": {
        "hf_dataset": "yelp_review_full",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _text_only("text"),
        "label_key": "label",
        "label_names": ["1", "2", "3", "4", "5"],
        "instruction": (
            "Rate the sentiment of the following Yelp review from 1 "
            "(most negative) to 5 (most positive)."
        ),
        "metric": "accuracy",
    },
    "dbpedia": {
        "hf_dataset": "fancyzhx/dbpedia_14",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _text_only("content"),
        "label_key": "label",
        "label_names": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building",
            "NaturalPlace", "Village", "Animal", "Plant", "Album",
            "Film", "WrittenWork",
        ],
        "instruction": (
            "Classify the following text into one of the categories: "
            "Company, EducationalInstitution, Artist, Athlete, "
            "OfficeHolder, MeanOfTransportation, Building, NaturalPlace, "
            "Village, Animal, Plant, Album, Film, WrittenWork."
        ),
        "metric": "accuracy",
    },
    "yahoo": {
        "hf_dataset": "yahoo_answers_topics",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _yahoo_builder,
        "label_key": "topic",
        "label_names": [
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government",
        ],
        "instruction": (
            "Classify the following question into one of the categories: "
            "Society & Culture, Science & Mathematics, Health, "
            "Education & Reference, Computers & Internet, Sports, "
            "Business & Finance, Entertainment & Music, "
            "Family & Relationships, Politics & Government."
        ),
        "metric": "accuracy",
    },
    # ---- GLUE tasks ----
    "mnli": {
        "hf_dataset": "nyu-mll/multi_nli",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "validation_matched",
        "input_builder": _nli_builder("premise", "hypothesis"),
        "label_key": "label",
        "label_names": ["entailment", "neutral", "contradiction"],
        "instruction": (
            "Determine the relationship between the premise and hypothesis. "
            "Answer with: entailment, neutral, or contradiction."
        ),
        "metric": "accuracy",
    },
    "qqp": {
        "hf_dataset": "glue",
        "hf_config": "qqp",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": lambda ex: (
            f"Question 1: {ex['question1']}\nQuestion 2: {ex['question2']}"
        ),
        "label_key": "label",
        "label_names": ["not_duplicate", "duplicate"],
        "instruction": (
            "Determine whether the following two questions are duplicates. "
            "Answer with: not_duplicate or duplicate."
        ),
        "metric": "accuracy",
    },
    "rte": {
        "hf_dataset": "super_glue",
        "hf_config": "rte",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _nli_builder("premise", "hypothesis"),
        "label_key": "label",
        "label_names": ["entailment", "not_entailment"],
        "instruction": (
            "Determine whether the premise entails the hypothesis. "
            "Answer with: entailment or not_entailment."
        ),
        "metric": "accuracy",
    },
    "sst-2": {
        "hf_dataset": "glue",
        "hf_config": "sst2",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _text_only("sentence"),
        "label_key": "label",
        "label_names": ["negative", "positive"],
        "instruction": (
            "Classify the sentiment of the following sentence. "
            "Answer with: negative or positive."
        ),
        "metric": "accuracy",
    },
    # ---- SuperGLUE tasks ----
    "wic": {
        "hf_dataset": "super_glue",
        "hf_config": "wic",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _wic_builder,
        "label_key": "label",
        "label_names": ["false", "true"],
        "instruction": (
            "Determine if the word is used with the same meaning in both "
            "sentences. Answer with: true or false."
        ),
        "metric": "accuracy",
    },
    "cb": {
        "hf_dataset": "super_glue",
        "hf_config": "cb",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _nli_builder("premise", "hypothesis"),
        "label_key": "label",
        "label_names": ["entailment", "contradiction", "neutral"],
        "instruction": (
            "Determine the relationship between the premise and hypothesis. "
            "Answer with: entailment, contradiction, or neutral."
        ),
        "metric": "accuracy",
    },
    "copa": {
        "hf_dataset": "super_glue",
        "hf_config": "copa",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _copa_builder,
        "label_key": "label",
        "label_names": ["choice1", "choice2"],
        "instruction": (
            "Choose the more plausible alternative. "
            "Answer with: choice1 or choice2."
        ),
        "metric": "accuracy",
    },
    "boolqa": {
        "hf_dataset": "super_glue",
        "hf_config": "boolq",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _boolq_builder,
        "label_key": "label",
        "label_names": ["false", "true"],
        "instruction": (
            "Answer the question based on the passage. "
            "Answer with: true or false."
        ),
        "metric": "accuracy",
    },
    "multirc": {
        "hf_dataset": "super_glue",
        "hf_config": "multirc",
        "hf_split_train": "train",
        "hf_split_val": "validation",
        "input_builder": _multirc_builder,
        "label_key": "label",
        "label_names": ["false", "true"],
        "instruction": (
            "Is the answer correct for the given question and passage? "
            "Answer with: true or false."
        ),
        "metric": "accuracy",
    },
    # ---- IMDB ----
    "imdb": {
        "hf_dataset": "imdb",
        "hf_config": None,
        "hf_split_train": "train",
        "hf_split_val": "test",
        "input_builder": _text_only("text"),
        "label_key": "label",
        "label_names": ["negative", "positive"],
        "instruction": (
            "Classify the sentiment of the following movie review. "
            "Answer with: negative or positive."
        ),
        "metric": "accuracy",
    },
}


# ---------------------------------------------------------------------------
# Task orders from the paper (Table 17)
# ---------------------------------------------------------------------------

TASK_ORDERS = {
    # Standard CL benchmark (5 tasks)
    "O1": ["yelp", "dbpedia", "amazon", "yahoo", "ag"],
    "O2": ["yelp", "dbpedia", "amazon", "ag", "yahoo"],
    "O3": ["yelp", "yahoo", "amazon", "ag", "dbpedia"],
    # Large Number of Tasks (15 tasks)
    "O4": [
        "mnli", "cb", "wic", "copa", "qqp", "boolqa", "rte", "imdb",
        "yelp", "amazon", "sst-2", "dbpedia", "ag", "multirc", "yahoo",
    ],
    "O5": [
        "multirc", "boolqa", "wic", "mnli", "cb", "copa", "qqp", "rte",
        "imdb", "sst-2", "dbpedia", "ag", "yelp", "amazon", "yahoo",
    ],
    "O6": [
        "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte", "imdb",
        "sst-2", "dbpedia", "ag", "yahoo", "multirc", "boolqa", "wic",
    ],
}


def get_task_order(order_name: str) -> list:
    """Return the task name list for a given order."""
    if order_name not in TASK_ORDERS:
        raise ValueError(
            f"Unknown order '{order_name}'. Available: {list(TASK_ORDERS.keys())}"
        )
    return TASK_ORDERS[order_name]


def get_task_config(task_name: str) -> dict:
    """Return the full config dict for a task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]
