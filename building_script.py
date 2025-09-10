import asyncio
import codecs
import copy
import json
import logging
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from typing import Any
from typing import Dict, List, Tuple

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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("kg_pipeline")


def secs(dt: float) -> str:
    return f"{dt:.3f}s"


def shorten(obj: Any, limit: int = 500) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s) - limit} chars)"


@contextmanager
def log_step(title: str, level: int = logging.INFO):
    t0 = time.time()
    logger.log(level, f"▶ {title} — start")
    try:
        yield
    except Exception as e:
        logger.exception(f"✖ {title} — failed: {e}")
        raise
    finally:
        logger.log(level, f"✔ {title} — done in {secs(time.time() - t0)}")


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


def initialize_abbreviation_subsystem(config_name: str) -> (
        Tuple)[Dict[str, Tuple[List[str], List[Tuple[str, str, str]]]], spacy.Language, MorphAnalyzer]:
    nlp = spacy.load('ru_core_news_sm')
    analyzer = MorphAnalyzer()
    with codecs.open(config_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'The abbreviation config "{config_name}" contains a wrong information!'
        raise ValueError(err_msg)
    keys = list(data.keys())
    abbr = dict()
    for k in keys:
        if not isinstance(k, str):
            err_msg = f'The abbreviation config "{config_name}" contains a wrong information!'
            raise ValueError(err_msg)
        if not k.isalpha():
            err_msg = f'The abbreviation config "{config_name}" contains a wrong information!'
            raise ValueError(err_msg)
        if not isinstance(data[k], str):
            err_msg = f'The abbreviation config "{config_name}" contains a wrong information!'
            raise ValueError(err_msg)
        v = data[k].strip()
        if len(v) == 0:
            err_msg = f'The abbreviation config "{config_name}" contains a wrong information!'
            raise ValueError(err_msg)
        doc = nlp(v)
        tokens = []
        morpho_data = []
        for token in doc:
            tokens.append(token.text)
            if str(token.pos_) in POS_DICT:
                pos = POS_DICT[str(token.pos_)]
            else:
                pos = str(token.pos_)
            case = token.morph.get('Case')
            if len(case) > 0:
                if len(case[0]) > 0:
                    case = CASE_DICT[str(case[0])]
                else:
                    case = ''
            else:
                case = ''
            number = token.morph.get('Number')
            if len(number) > 0:
                if len(number[0]) > 0:
                    number = NUMBER_DICT[str(number[0])]
                else:
                    number = ''
            else:
                number = ''
            morpho_data.append((pos, case, number))
        abbr[k.lower()] = (tokens, morpho_data)
        del doc, tokens, morpho_data
    return abbr, nlp, analyzer


def tokenize_and_analyze_morphology(text: str, nlp: spacy.Language) -> (
        Tuple)[List[str], List[Tuple[int, int]], List[Tuple[str, str, str]]]:
    doc = nlp(text)
    all_tokens = []
    all_bounds = []
    all_morpho = []
    for token in doc:
        all_tokens.append(token.text)
        all_bounds.append((token.idx, token.idx + len(token)))
        pos = POS_DICT.get(str(token.pos_), str(token.pos_))
        case = token.morph.get('Case')
        if len(case) > 0:
            if len(case[0]) > 0:
                case = CASE_DICT[str(case[0])]
                if case == 'accs':
                    case = 'loct'
            else:
                case = ''
        else:
            case = ''
        number = token.morph.get('Number')
        if len(number) > 0:
            if len(number[0]) > 0:
                number = NUMBER_DICT[str(number[0])]
            else:
                number = ''
        else:
            number = ''
        all_morpho.append((pos, case, number))
    del doc
    return all_tokens, all_bounds, all_morpho


def find_main_token(phrase_tokens: List[str], morpho: List[Tuple[str, str, str]]) -> int:
    found_idx = -1
    n = len(phrase_tokens)
    if n != len(morpho):
        err_msg = (f'Number of tokens does not equal to number of morphological items! '
                   f'{phrase_tokens} != {morpho}')
        raise ValueError(err_msg)
    for idx in range(n):
        if (morpho[idx][0] in {'NOUN', 'NPRO'}) and (morpho[idx][1] == 'nomn'):
            found_idx = idx
    return found_idx


def find_form(token: str, morpho: Tuple[str, str, str], analyzer: MorphAnalyzer) -> Parse:
    variants = analyzer.parse(token)
    best = variants[0]
    for it in variants[1:]:
        if it.tag.POS == morpho[0] and (morpho[2] in {'sing', 'plur'} and it.tag.number == morpho[2]):
            best = it
    return best


def inflect_phrase(phrase_tokens: List[str], morpho: List[Tuple[str, str, str]], inflector: MorphAnalyzer,
                   target_case: str, target_number: str) -> str:
    main_token_idx = find_main_token(phrase_tokens, morpho)
    if main_token_idx < 0:
        warnings.warn(f'The text "{" ".join(phrase_tokens)}" cannot be inflected!')
        return ' '.join(phrase_tokens)
    inflected_tokens = []
    for idx, val in enumerate(phrase_tokens):
        if idx <= main_token_idx:
            # Исключаем пустые значения
            target_grammemes = set()
            if target_case:
                target_grammemes.add(target_case)
            if target_number:
                target_grammemes.add(target_number)

            inflection = find_form(val, morpho[idx], inflector).inflect(target_grammemes)
            if inflection is None:
                inflected_tokens.append(val)
            else:
                inflected_tokens.append(inflection.word)
        else:
            inflected_tokens.append(val)
    if len(inflected_tokens) == 0:
        return ''
    if inflected_tokens[0][0].islower():
        inflected_tokens[0] = inflected_tokens[0][0].upper() + inflected_tokens[0][1:]
    return ' '.join(inflected_tokens)


def replace_abbreviations(old_text: str, abbreviation_config: Dict[str, Tuple[List[str], List[Tuple[str, str, str]]]],
                          nlp: spacy.Language, morph: MorphAnalyzer) -> str:
    tokens, bounds, morpho_data = tokenize_and_analyze_morphology(old_text, nlp)
    new_text = copy.copy(old_text)
    for idx, (token, (pos, case, number)) in enumerate(zip(tokens, morpho_data)):
        if token.lower() in abbreviation_config:
            case = case if case in CASE_DICT.values() else ''
            number = number if number in NUMBER_DICT.values() else ''
            new_token = inflect_phrase(
                abbreviation_config[token.lower()][0],
                abbreviation_config[token.lower()][1],
                morph,
                case,
                number
            )
            new_text = new_text[:bounds[idx][0]] + new_token + new_text[bounds[idx][1]:]
            if len(new_token) != len(token):
                bounds[idx] = (
                    bounds[idx][0],
                    bounds[idx][1] + len(new_token) - len(token)
                )
                for other_idx in range(idx + 1, len(bounds)):
                    bounds[other_idx] = (
                        bounds[other_idx][0] + len(new_token) - len(token),
                        bounds[other_idx][1] + len(new_token) - len(token)
                    )
    return new_text


def convert(text: str) -> str:
    return replace_abbreviations(text, config, spacy, pymorphy)


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


async def gte_hf_embed(texts: List[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    encoded_texts = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True
    ).to(device)
    batch_dict = tokenizer(
        texts, return_tensors='pt',
        max_length=LOCAL_EMBEDDER_MAX_TOKENS, padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = embed_model(**batch_dict)
        embeddings = F.normalize(
            outputs.last_hidden_state[:, 0][:LOCAL_EMBEDDER_DIMENSION],
            p=2, dim=1
        )
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()


def explain_abbreviations_with_llm(document: str, abbreviations: dict) -> str:
    snow_stemmer = SnowballStemmer(language='russian')
    filtered_abbreviations = dict()
    for cur_word in nltk.wordpunct_tokenize(document):
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
        return document
    user_prompt = TEMPLATE_FOR_ABBREVIATION_EXPLAINING.format(
        abbreviations_dict=filtered_abbreviations,
        text_of_document=document,
        special_masks=special_token_keys
    )
    messages = [{'role': 'user', 'content': user_prompt}]
    global client
    try:
        response = client.chat.completions.create(
            model=LLM_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            n=1,
            max_tokens=QUERY_MAX_TOKENS
        )
        new_improved_document = response.choices[0].message.content
        del response
    except:
        new_improved_document = document
    del messages, user_prompt
    return new_improved_document


async def initialize_rag():
    logger.info("Инициализация эмбеддера и RAG...")
    with log_step("Загрузка локального эмбеддера"):
        emb_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_EMBEDDER_NAME
        )
        emb_model = AutoModel.from_pretrained(
            LOCAL_EMBEDDER_NAME,
            trust_remote_code=True
        )
        emb_model.eval()
    with log_step("Создание LightRAG"):
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

    with log_step("Инициализация хранилищ RAG"):
        await rag.initialize_storages()
        await initialize_pipeline_status()
    logger.info("LightRAG готов.")
    return rag


if __name__ == '__main__':
    setup_logger("lightrag", level="INFO")
    random.seed(RANDOM_SEED)
    logger.info("Скрипт запущен. RANDOM_SEED=%s", RANDOM_SEED)

    dataset_dir = 'pages_txt'
    print(f'os.path.isdir({dataset_dir}) = {os.path.isdir(dataset_dir)}')

    text_files = list(
        map(lambda it2: os.path.join(dataset_dir, it2),
            filter(lambda it1: it1.endswith('.txt'), os.listdir(dataset_dir))))
    # python -m spacy download ru_core_news_sm
    config_fname = os.path.join('full_abbreviations_updated.json')
    config, spacy, pymorphy = initialize_abbreviation_subsystem(config_fname)

    text_data = []
    for cur_fname in text_files:
        with codecs.open(cur_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
            new_text = '\n'.join(list(map(
                lambda it3: ' '.join(it3.replace('\r', ' ').split()),
                filter(
                    lambda it2: len(it2) > 0,
                    map(
                        lambda it1: it1.strip(),
                        fp.readlines()
                    )
                )
            ))).strip()
        if len(new_text) > 0:
            text_data.append(convert(new_text))
        del new_text

    logger.info("Загружено из %s: %d txt-файлов", dataset_dir, len(text_data))
    print(f'3 random documents:')
    for it in random.sample(text_data, 3):
        print('\n' + it)

    special_tokens = dict()
    re_for_special_tokens = re.compile(r'\[\w+?\]')
    for cur_text in tqdm(text_data):
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
    print(f'There are {len(special_tokens)} special tokens. They are:')
    for it in special_token_keys:
        print('{0:>20}: {1:>6}'.format(it, special_tokens[it]))

    WORKING_DIR = 'prepared_it_new'
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    VLLM_API_KEY = ''
    VLLM_BASE_URL = "http://127.0.0.1:9111/v1"
    os.environ['OPENAI_API_KEY'] = VLLM_API_KEY
    LLM_NAME = 'meno-medium-0.1'
    TEMPERATURE = 0.3
    QUERY_MAX_TOKENS = 8000
    LOCAL_EMBEDDER_DIMENSION = 768
    LOCAL_EMBEDDER_MAX_TOKENS = 4096
    LOCAL_EMBEDDER_NAME = '/workspace/data/models/gte-multilingual-base'
    logger.info(f'os.path.isdir({LOCAL_EMBEDDER_NAME}) = {os.path.isdir(LOCAL_EMBEDDER_NAME)}')
    ABBREVIATIONS_FNAME = 'full_abbreviations_updated.json'
    logger.info(f'os.path.isfile({ABBREVIATIONS_FNAME}) = {os.path.isfile(ABBREVIATIONS_FNAME)}')
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
    logger.info("WORKING_DIR=%s", WORKING_DIR)
    logger.info("DATASET_DIR=%s (exists=%s)", dataset_dir, os.path.isdir(dataset_dir))
    logger.info("EMBEDDER_PATH=%s (exists=%s)", LOCAL_EMBEDDER_NAME, os.path.isdir(LOCAL_EMBEDDER_NAME))
    logger.info("ABBR_JSON=%s (exists=%s)", ABBREVIATIONS_FNAME, os.path.isfile(ABBREVIATIONS_FNAME))
    logger.info("LLM: model=%s, base_url=%s, temp=%.2f, max_tokens=%d", LLM_NAME, VLLM_BASE_URL, TEMPERATURE,
                QUERY_MAX_TOKENS)

    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    )

    MAX_NUMBER_OF_TEXTS = None

    improved_texts_fname = os.path.join(WORKING_DIR, 'improved_texts.json')

    if os.path.isfile(improved_texts_fname):
        with open(improved_texts_fname, mode='r', encoding='utf-8') as fp:
            improved_texts = json.load(fp)
        logger.info("Загружено improved_texts: %d", len(improved_texts))

    data_iter = text_data if MAX_NUMBER_OF_TEXTS is None else text_data[:MAX_NUMBER_OF_TEXTS]
    for cur_text in tqdm(data_iter):
        improved_texts.append(cur_text)

    with codecs.open(improved_texts_fname, mode='w', encoding='utf-8') as fp:
        json.dump(fp=fp, obj=improved_texts, ensure_ascii=False, indent=4)
        logger.info("Сохранено improved_texts.json (%d)", len(improved_texts))

    # print(f'3 random examples of improved texts:')
    # if MAX_NUMBER_OF_TEXTS is None:
    #     selected_indices = random.sample(list(range(len(text_data))), 3)
    # else:
    #     selected_indices = random.sample(list(range(len(text_data[0:MAX_NUMBER_OF_TEXTS]))), 3)
    # for example_index in selected_indices:
    #     print('')
    #     print('BEFORE IMPROVING:')
    #     print(' '.join(text_data[example_index].split()))
    #     print('AFTER IMPROVING:')
    #     print(' '.join(improved_texts[example_index].split()))

    nest_asyncio.apply()
    with log_step("Инициализация RAG (async)"):
        rag = asyncio.run(initialize_rag())
    with log_step("Вставка документов в хранилища RAG"):
        if MAX_NUMBER_OF_TEXTS is None:
            for cur_text in tqdm(improved_texts):
                rag.insert(cur_text)
        else:
            for cur_text in tqdm(improved_texts[0:MAX_NUMBER_OF_TEXTS]):
                rag.insert(cur_text)
    with log_step("Финализация RAG-хранилищ"):
        rag.finalize_storages()
    logger.info("Пайплайн завершён успешно.")
