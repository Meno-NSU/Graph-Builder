import asyncio
import codecs
import copy
import json
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Tuple, Any, Optional

import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("abbr_rag_pipeline")


def secs(dt: float) -> str:
    return f"{dt:.3f}s"


def shorten(obj: Any, limit: int = 200) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s) - limit} chars)"


def safe_len(x) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


def log_exception(e: Exception) -> None:
    logger.exception("Exception: %s", e)


def log_kv(**kw):
    return " ".join(f"{k}={kw[k]}" for k in kw)


def debug_mem(prefix: str = ""):
    try:
        import psutil, os
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss / (1024 ** 2)
        logger.debug("%sRSS=%.1f MB", f"{prefix} " if prefix else "", rss)
    except Exception:
        pass


def time_now() -> float:
    return time.perf_counter()


def elapsed(t0: float) -> str:
    return secs(time_now() - t0)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def log_iter_preview(name: str, it, n: int = 3):
    try:
        lst = list(it)
        logger.debug("%s: total=%d sample=%s", name, len(lst), shorten(lst[:n]))
        return lst
    except Exception as e:
        logger.warning("Failed preview for %s: %s", name, e)
        return it


def device_info():
    try:
        import torch
        cuda = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0) if cuda and device_count > 0 else "CPU"
        logger.info("Torch device: cuda=%s device_count=%d name=%s", cuda, device_count, dev_name)
    except Exception as e:
        logger.warning("Torch not available or failed to query device: %s", e)


def log_io(level=logging.DEBUG):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time_now()
            try:
                logger.log(level, "→ %s(args=%s, kwargs=%s)", fn.__name__, shorten(args), shorten(kwargs))
                res = fn(*args, **kwargs)
                logger.log(level, "← %s done in %s; result_preview=%s",
                           fn.__name__, elapsed(t0), shorten(res, 300))
                return res
            except Exception as e:
                logger.error("✖ %s failed in %s: %s", fn.__name__, elapsed(t0), e)
                log_exception(e)
                raise

        return wrapper

    return deco


@contextmanager
def log_section(title: str, level=logging.INFO):
    t0 = time_now()
    logger.log(level, "▶ %s ...", title)
    try:
        yield
        logger.log(level, "■ %s done in %s", title, elapsed(t0))
    except Exception as e:
        logger.error("■ %s failed in %s: %s", title, elapsed(t0), e)
        log_exception(e)
        raise


import nest_asyncio
import nltk
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from nltk.stem.snowball import SnowballStemmer
from openai import OpenAI
from pymorphy3 import MorphAnalyzer
from pymorphy3.analyzer import Parse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

RANDOM_SEED = 42

POS_DICT: Dict[str, str] = {
    'NOUN': 'NOUN',
    'PROPN': 'NOUN',
    'PRON': 'NPRO',
    'DET': 'NPRO',
    'ADJ': 'ADJF',
    'VERB': 'VERB',
    'AUX': 'PRCL',
    'PART': 'PRCL',
    'ADV': 'ADVB',
    'NUM': 'NUMR',
    'ADP': 'PREP',
    'CCONJ': 'CONJ',
    'SCONJ': 'CONJ',
    'INTJ': 'INTJ'
}

CASE_DICT: Dict[str, str] = {
    'Acc': 'accs',
    'Dat': 'datv',
    'Gen': 'gent',
    'Ins': 'ablt',
    'Loc': 'loct',
    'Nom': 'nomn',
    'Par': 'gen2',
    'Voc': 'voct'
}

NUMBER_DICT: Dict[str, str] = {
    'Sing': 'sing',
    'Plur': 'plur'
}


@log_io()
def initialize_abbreviation_subsystem(config_name: str) -> Tuple[
    Dict[str, Tuple[List[str], List[Tuple[str, str, str]]]],
    spacy.Language, MorphAnalyzer
]:
    with log_section(f"Load spaCy model 'ru_core_news_sm'"):
        nlp = spacy.load('ru_core_news_sm')
    with log_section("Init pymorphy3 MorphAnalyzer"):
        analyzer = MorphAnalyzer()
    with log_section(f"Read abbreviation config: {config_name}"):
        with codecs.open(config_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f'Abbreviation config "{config_name}" must be a dict')
    logger.info("Abbreviation entries: %d", len(data))
    abbr = dict()
    for k, v in data.items():
        if not isinstance(k, str) or not k.isalpha() or not isinstance(v, str) or not v.strip():
            raise ValueError(f'Config "{config_name}" contains invalid key/value: {k} -> {v}')
        v = v.strip()
        doc = nlp(v)
        tokens, morpho_data = [], []
        for token in doc:
            tokens.append(token.text)
            pos = POS_DICT.get(str(token.pos_), str(token.pos_))
            case_list = token.morph.get('Case')
            case = CASE_DICT.get(case_list[0], '') if case_list and case_list[0] else ''
            number_list = token.morph.get('Number')
            number = NUMBER_DICT.get(number_list[0], '') if number_list and number_list[0] else ''
            morpho_data.append((pos, case, number))
        abbr[k.lower()] = (tokens, morpho_data)
    logger.info("Built abbreviation map: %d items", len(abbr))
    return abbr, nlp, analyzer


@log_io()
def tokenize_and_analyze_morphology(text: str, nlp: spacy.Language) -> Tuple[
    List[str], List[Tuple[int, int]], List[Tuple[str, str, str]]
]:
    doc = nlp(text)
    all_tokens, all_bounds, all_morpho = [], [], []
    for token in doc:
        all_tokens.append(token.text)
        all_bounds.append((token.idx, token.idx + len(token)))
        pos = POS_DICT.get(str(token.pos_), str(token.pos_))
        case_list = token.morph.get('Case')
        if case_list and case_list[0]:
            case = CASE_DICT.get(case_list[0], '')
            if case == 'accs':
                case = 'loct'
        else:
            case = ''
        number_list = token.morph.get('Number')
        number = NUMBER_DICT.get(number_list[0], '') if number_list and number_list[0] else ''
        all_morpho.append((pos, case, number))
    logger.debug("Tokenized: tokens=%d bounds=%d morpho=%d",
                 len(all_tokens), len(all_bounds), len(all_morpho))
    return all_tokens, all_bounds, all_morpho


@log_io()
def find_main_token(phrase_tokens: List[str], morpho: List[Tuple[str, str, str]]) -> int:
    if len(phrase_tokens) != len(morpho):
        raise ValueError(f'Mismatch tokens vs morpho: {len(phrase_tokens)} != {len(morpho)}')
    found_idx = -1
    for idx, m in enumerate(morpho):
        if (m[0] in {'NOUN', 'NPRO'}) and (m[1] == 'nomn'):
            found_idx = idx
    logger.debug("find_main_token: idx=%d phrase=%s", found_idx, shorten(phrase_tokens))
    return found_idx


@log_io()
def find_form(token: str, morpho: Tuple[str, str, str], analyzer: MorphAnalyzer) -> Parse:
    variants = analyzer.parse(token)
    if not variants:
        raise ValueError(f"No morph variants returned for token={token}")
    best = variants[0]
    for it in variants[1:]:
        try:
            if it.tag.POS == morpho[0] and (morpho[2] in {'sing', 'plur'} and it.tag.number == morpho[2]):
                best = it
        except Exception:
            pass
    return best


@log_io()
def inflect_phrase(
        phrase_tokens: List[str],
        morpho: List[Tuple[str, str, str]],
        inflector: MorphAnalyzer,
        target_case: str,
        target_number: str
) -> str:
    main_token_idx = find_main_token(phrase_tokens, morpho)
    if main_token_idx < 0:
        msg = f'The text "{" ".join(phrase_tokens)}" cannot be inflected!'
        warnings.warn(msg)
        logger.warning(msg)
        return ' '.join(phrase_tokens)
    inflected_tokens = []
    for idx, val in enumerate(phrase_tokens):
        if idx <= main_token_idx:
            target_grammemes = set()
            if target_case:
                target_grammemes.add(target_case)
            if target_number:
                target_grammemes.add(target_number)
            try:
                inflection = find_form(val, morpho[idx], inflector).inflect(target_grammemes)
                inflected_tokens.append(inflection.word if inflection is not None else val)
            except Exception as e:
                logger.debug("Inflect fail token=%s grams=%s: %s", val, target_grammemes, e)
                inflected_tokens.append(val)
        else:
            inflected_tokens.append(val)
    if not inflected_tokens:
        return ''
    if inflected_tokens[0] and inflected_tokens[0][0].islower():
        inflected_tokens[0] = inflected_tokens[0][0].upper() + inflected_tokens[0][1:]
    res = ' '.join(inflected_tokens)
    logger.debug("Inflected: %s -> %s", shorten(' '.join(phrase_tokens)), shorten(res))
    return res


@log_io()
def replace_abbreviations(
        old_text: str,
        abbreviation_config: Dict[str, Tuple[List[str], List[Tuple[str, str, str]]]],
        nlp: spacy.Language,
        morph: MorphAnalyzer
) -> str:
    tokens, bounds, morpho_data = tokenize_and_analyze_morphology(old_text, nlp)
    new_text = copy.copy(old_text)
    replaced = 0
    for idx, (token, (pos, case, number)) in enumerate(zip(tokens, morpho_data)):
        key = token.lower()
        if key in abbreviation_config:
            case = case if case in CASE_DICT.values() else ''
            number = number if number in NUMBER_DICT.values() else ''
            new_token = inflect_phrase(
                abbreviation_config[key][0],
                abbreviation_config[key][1],
                morph,
                case,
                number
            )
            new_text = new_text[:bounds[idx][0]] + new_token + new_text[bounds[idx][1]:]
            delta = len(new_token) - len(token)
            if delta != 0:
                bounds[idx] = (bounds[idx][0], bounds[idx][1] + delta)
                for other_idx in range(idx + 1, len(bounds)):
                    bounds[other_idx] = (bounds[other_idx][0] + delta, bounds[other_idx][1] + delta)
            replaced += 1
    logger.info("Abbreviation replacements done: %d", replaced)
    return new_text


def convert(text: str) -> str:
    t0 = time_now()
    logger.debug("convert: in_len=%d", len(text))
    new_text = replace_abbreviations(text, config, spacy_nlp, pymorphy_an)
    logger.debug("convert: out_len=%d elapsed=%s", len(new_text), elapsed(t0))
    return new_text


@log_io()
async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_NAME,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        temperature=TEMPERATURE,
        **kwargs
    )


@log_io()
async def gte_hf_embed(texts: List[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    logger.debug("Embed batch: size=%d device=%s", len(texts), device)
    batch_dict = tokenizer(
        texts, return_tensors='pt',
        max_length=LOCAL_EMBEDDER_MAX_TOKENS, padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = embed_model(**batch_dict)
        cls = outputs.last_hidden_state[:, 0, :]  # [B, H]
        if cls.shape[1] > LOCAL_EMBEDDER_DIMENSION:
            cls = cls[:, :LOCAL_EMBEDDER_DIMENSION]
        embeddings = F.normalize(cls, p=2, dim=1)
    if embeddings.dtype == torch.bfloat16:
        arr = embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        arr = embeddings.detach().cpu().numpy()
    logger.debug("Embed out: shape=%s dtype=%s", arr.shape, arr.dtype)
    debug_mem("after-embed")
    return arr


@log_io()
def explain_abbreviations_with_llm(document: str, abbreviations: dict) -> str:
    snow_stemmer = SnowballStemmer(language='russian')
    filtered_abbreviations = dict()
    tokens = nltk.wordpunct_tokenize(document)
    for cur_word in tokens:
        if cur_word in abbreviations:
            filtered_abbreviations[cur_word] = abbreviations[cur_word]
        elif cur_word.lower() in abbreviations:
            filtered_abbreviations[cur_word] = abbreviations[cur_word.lower()]
        elif cur_word.upper() in abbreviations:
            filtered_abbreviations[cur_word] = abbreviations[cur_word.upper()]
        else:
            stem = snow_stemmer.stem(cur_word)
            if stem in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[stem]
            elif stem.lower() in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[stem.lower()]
            elif stem.upper() in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[stem.upper()]
    del snow_stemmer
    if len(filtered_abbreviations) == 0:
        logger.info("No abbreviations detected by stemmer — skipping LLM rewrite")
        return document
    user_prompt = TEMPLATE_FOR_ABBREVIATION_EXPLAINING.format(
        abbreviations_dict=filtered_abbreviations,
        text_of_document=document,
        special_masks=special_token_keys
    )
    messages = [{'role': 'user', 'content': user_prompt}]
    global client
    try:
        logger.debug("LLM call: model=%s prompt_len=%d", LLM_NAME, len(user_prompt))
        response = client.chat.completions.create(
            model=LLM_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            n=1,
            max_tokens=QUERY_MAX_TOKENS
        )
        new_improved_document = response.choices[0].message.content
        logger.info("LLM response received: len=%d", len(new_improved_document) if new_improved_document else -1)
        del response
    except Exception as e:
        logger.error("LLM call failed, fallback to original text: %s", e)
        new_improved_document = document
    return new_improved_document


@log_io()
async def initialize_rag():
    with log_section("Load local embedding tokenizer/model"):
        emb_tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBEDDER_NAME)
        emb_model = AutoModel.from_pretrained(LOCAL_EMBEDDER_NAME, trust_remote_code=True)
        emb_model.eval()
    device_info()

    with log_section("Initialize LightRAG"):
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            cosine_better_than_threshold=0.1,
            embedding_func=EmbeddingFunc(
                embedding_dim=LOCAL_EMBEDDER_DIMENSION,
                max_token_size=LOCAL_EMBEDDER_MAX_TOKENS,
                func=lambda texts: gte_hf_embed(
                    texts,
                    tokenizer=emb_tokenizer,
                    embed_model=emb_model
                )
            ),
            addon_params={'language': 'Russian'},
        )

    with log_section("Initialize storages & pipeline status"):
        await rag.initialize_storages()
        await initialize_pipeline_status()

    return rag


if __name__ == '__main__':
    setup_logger("lightrag", level="INFO")
    random.seed(RANDOM_SEED)
    debug_mem("start")

    dataset_dir = 'pages_txt'
    WORKING_DIR = 'prepared_it_new'
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "everest.nsu.ru:9111")
    os.environ['OPENAI_API_KEY'] = VLLM_API_KEY
    LLM_NAME = 'RuadaptQwen3-32B-Instruct'
    TEMPERATURE = 0.3
    QUERY_MAX_TOKENS = 8000
    LOCAL_EMBEDDER_DIMENSION = 768
    LOCAL_EMBEDDER_MAX_TOKENS = 4096
    LOCAL_EMBEDDER_NAME = '/workspace/data/models/gte-multilingual-base'
    ABBREVIATIONS_FNAME = 'full_abbreviations_updated.json'
    MAX_NUMBER_OF_TEXTS = None
    TEMPLATE_FOR_ABBREVIATION_EXPLAINING = '''Отредактируйте, пожалуйста, текст заданного документа так, чтобы этот документ стал более простым и понятным для обычных людей от юных старшеклассников до пожилых мужчин и женщин. При этом не надо, пожалуйста, применять markdown или иной вид гипертекста. Главное, на что вам надо обратить внимание и по возможности исправить - это логика изложения и понятность формулировок документа. Ничего не объясняйте и не комментируйте своё решение, просто перепишите текст документа.

    Обратите внимание, что документ анонимизирован, то есть все именованные сущности заменены специальными словами-масками в квадратных скобках (например, вместо текста "Иван Иванович Иванов любит горчицу" вы встретите текст "[NAME] любит горчицу"). Полный список специальных слов-масок приведён здесь: {special_masks}. Не изменяйте этих слов, пожалуйста, а оставляйте как есть.

    Также исправьте грамматические ошибки в тексте документа, если они там есть. Кроме того, если вы обнаружите аббревиатуры в тексте этого документа, то замените все обнаруженные аббревиатуры их корректными расшифровками, сохранив морфологическую и синтаксическую согласованность. Вот здесь вы можете ознакомиться с JSON-словарём, описывающим возможные аббревиатуры и их расшифровки:

    ```json
    {abbreviations_dict}
    ```

    Далее приведён текст документа, нуждающийся в возможном улучшении:

    ```text
    {text_of_document}
    ```'''

    logger.info("ENV " + log_kv(
        dataset_dir=dataset_dir,
        WORKING_DIR=WORKING_DIR,
        LLM_NAME=LLM_NAME,
        LOCAL_EMBEDDER_NAME=LOCAL_EMBEDDER_NAME,
        LOCAL_EMBEDDER_DIMENSION=LOCAL_EMBEDDER_DIMENSION,
        LOCAL_EMBEDDER_MAX_TOKENS=LOCAL_EMBEDDER_MAX_TOKENS,
        VLLM_BASE_URL=VLLM_BASE_URL,
        KEY_SET=bool(VLLM_API_KEY)
    ))

    with log_section("Sanity checks"):
        logger.info("Check dataset dir exists: %s -> %s", dataset_dir, os.path.isdir(dataset_dir))
        logger.info("Check local embedder dir exists: %s -> %s", LOCAL_EMBEDDER_NAME,
                    os.path.isdir(LOCAL_EMBEDDER_NAME))
        logger.info("Check abbreviations file exists: %s -> %s", ABBREVIATIONS_FNAME,
                    os.path.isfile(ABBREVIATIONS_FNAME))

    with log_section("Collect .txt files"):
        if not os.path.isdir(dataset_dir):
            logger.error("Dataset dir does not exist: %s", dataset_dir)
        text_files = [
            os.path.join(dataset_dir, fname)
            for fname in os.listdir(dataset_dir) if fname.endswith(".txt")
        ]
        logger.info("Text files found: %d", len(text_files))
        logger.debug("Sample files: %s", shorten(text_files[:5], 500))

    config_fname = os.path.join(ABBREVIATIONS_FNAME)
    with log_section("Initialize abbreviation subsystem"):
        config, spacy_nlp, pymorphy_an = initialize_abbreviation_subsystem(config_fname)

    text_data: List[str] = []
    with log_section("Read & normalize text files"):
        for cur_fname in tqdm(text_files, desc="read_files"):
            try:
                with codecs.open(cur_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
                    lines = fp.readlines()
                new_text = '\n'.join(
                    ' '.join(line.replace('\r', ' ').split())
                    for line in (ln.strip() for ln in lines)
                    if len(line) > 0
                ).strip()
                if new_text:
                    text_data.append(convert(new_text))
            except Exception as e:
                logger.error("Failed to read/process file %s: %s", cur_fname, e)
    logger.info("Number of documents: %d", len(text_data))

    if text_data:
        try:
            logger.info("3 random documents preview:")
            for it in random.sample(text_data, min(3, len(text_data))):
                logger.debug("DOC:\n%s\n", shorten(it, 2000))
        except Exception:
            pass

    with log_section("Scan special tokens like [MASK]"):
        special_tokens: Dict[str, int] = dict()
        re_for_special_tokens = re.compile(r'\[\w+?\]')
        for cur_text in tqdm(text_data, desc="scan_special_tokens"):
            start_pos = 0
            search_res = re_for_special_tokens.search(cur_text[start_pos:])
            while search_res is not None:
                token_start = start_pos + search_res.start()
                token_end = start_pos + search_res.end()
                new_special_token = cur_text[token_start:token_end]
                special_tokens[new_special_token] = special_tokens.get(new_special_token, 0) + 1
                start_pos = token_end
                search_res = re_for_special_tokens.search(cur_text[start_pos:])
        special_token_keys = sorted(list(special_tokens.keys()), key=lambda it: (-special_tokens[it], it))
        logger.info("Special tokens: unique=%d total=%d",
                    len(special_tokens), sum(special_tokens.values()))
        for it in special_token_keys[:50]:
            logger.debug("%20s : %6d", it, special_tokens[it])

    with log_section("Prepare working dir"):
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
            logger.info("Created dir: %s", WORKING_DIR)
        else:
            logger.info("Dir exists: %s", WORKING_DIR)

    with log_section("Init OpenAI client"):
        try:
            client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
            logger.info("OpenAI client initialized (base_url=%s, key_set=%s)",
                        VLLM_BASE_URL, bool(VLLM_API_KEY))
        except Exception as e:
            logger.error("Failed to init OpenAI client: %s", e)
            client = None

    improved_texts_fname = os.path.join(WORKING_DIR, 'improved_texts.json')
    improved_texts: List[str] = []

    with log_section("Load or build improved_texts"):
        if os.path.isfile(improved_texts_fname):
            logger.info("Found existing improved_texts: %s", improved_texts_fname)
            try:
                with open(improved_texts_fname, mode='r', encoding='utf-8') as fp:
                    improved_texts = json.load(fp)
                logger.info("Loaded improved_texts: %d items", len(improved_texts))
            except Exception as e:
                logger.error("Failed to read existing improved_texts, will rebuild: %s", e)

        if not improved_texts:
            logger.warning("No improved_texts present — fallback to original text_data (no LLM rewrite).")
            improved_texts = list(text_data)

        try:
            with codecs.open(improved_texts_fname, mode='w', encoding='utf-8') as fp:
                json.dump(improved_texts, fp, ensure_ascii=False, indent=2)  # FIX: порядок аргументов
            logger.info("Saved improved_texts to %s (count=%d)", improved_texts_fname, len(improved_texts))
        except Exception as e:
            logger.error("Failed to save improved_texts: %s", e)

    with log_section("Show 3 examples of improved_texts"):
        if improved_texts:
            if MAX_NUMBER_OF_TEXTS is None:
                pool = list(range(len(text_data)))
            else:
                pool = list(range(min(len(text_data), MAX_NUMBER_OF_TEXTS)))
            if pool:
                selected_indices = random.sample(pool, min(3, len(pool)))
                for idx in selected_indices:
                    logger.debug("BEFORE:\n%s", shorten(' '.join(text_data[idx].split()), 2000))
                    logger.debug("AFTER:\n%s", shorten(' '.join(improved_texts[idx].split()), 2000))
        else:
            logger.warning("improved_texts is empty — nothing to preview")

    nest_asyncio.apply()
    with log_section("Initialize RAG (async)"):
        try:
            rag = asyncio.run(initialize_rag())
        except Exception as e:
            logger.error("RAG init failed: %s", e)
            raise

    with log_section("Insert documents into RAG"):
        to_iter = improved_texts if MAX_NUMBER_OF_TEXTS is None else improved_texts[:MAX_NUMBER_OF_TEXTS]
        for cur_text in tqdm(to_iter, desc="rag_insert"):
            try:
                rag.insert(cur_text)
            except Exception as e:
                logger.error("Failed to insert doc into RAG: %s", e)

    with log_section("Finalize storages"):
        try:
            rag.finalize_storages()
        except Exception as e:
            logger.error("Finalize storages failed: %s", e)

    logger.info("Pipeline complete.")
    debug_mem("end")
