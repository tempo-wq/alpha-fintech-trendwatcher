import json
import streamlit as st
from typing import List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import GOLDEN_DATASET

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

# --- СХЕМА ДАННЫХ (Pydantic) ---
class SignalCard(BaseModel):
    is_noise: bool = Field(..., description="True, если публикация не имеет прямого отношения к финтех-индустрии.")
    headline: str = Field(..., description="Краткий аналитический заголовок зафиксированного сигнала.")
    llm_hotness: int = Field(..., description="Первичная оценка значимости события от 1 до 5 по шкале LLM.")
    why_now: str = Field(..., description="Профессиональное обоснование актуальности: почему данное событие критично именно сейчас.")
    category: str = Field(..., description="Категориальная принадлежность: банковский продукт / платёжный сервис / регулирование / рынок.")
    sources: List[str] = Field(..., description="Верифицированные ссылки на первоисточники.")
    summary: str = Field(..., description="Краткая выжимка сути происходящего (в пределах 2-3 предложений).")
    draft: str = Field(..., description="Черновик информационного материала для внутренней команды.")

# --- СЕМАНТИЧЕСКАЯ ДЕДУПЛИКАЦИЯ ---
def deduplicate_articles_semantic(articles: list, threshold: float) -> list:
    if not articles: return []
    texts = [article['text'] for article in articles]
    embeddings = embedding_model.encode(texts)
    
    unique_articles = []
    seen_indices = set()
    
    for i in range(len(articles)):
        if i in seen_indices: continue
        current_article = articles[i].copy()
        current_article['extra_sources'] = []
        current_article['merge_scores'] = [] 
        
        for j in range(i + 1, len(articles)):
            if j in seen_indices: continue
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            
            if similarity >= threshold:
                current_article['extra_sources'].append(articles[j]['url'])
                current_article['merge_scores'].append(round(float(similarity), 2))
                seen_indices.add(j)
                
        unique_articles.append(current_article)
        seen_indices.add(i)
        
    return unique_articles

# --- ГИБРИДНЫЙ СКОРИНГ ---
def calculate_hybrid_hotness(llm_score: int, text: str, sources: List[str]) -> int:
    # Базовый вес от LLM
    score = llm_score * 0.4
    
    # НОВОЕ: Влияние количества дублей (N_clusters)
    # Считаем количество дополнительных источников (дублей)
    num_duplicates = max(0, len(sources) - 1)
    # Начисляем по 0.3 балла за каждый дубль, но не более 1.5 баллов суммарно, 
    # чтобы массовый репост не сломал общую логику оценки
    score += min(num_duplicates * 0.3, 1.5)
    
    # Трастовость источника (I_source)
    high_trust = ['cbr.ru', 'vedomosti.ru', 'kommersant.ru']
    if any(any(ht in s for s in sources) for ht in high_trust):
        score += 1.5
    elif any('banki.ru' in s for s in sources):
        score += 1.0
        
    # Триггерные слова (T_trigger)
    triggers = ['санкции', 'ставка', 'регулирование', 'цифровой рубль', 'запрет', 'ключевая']
    if any(t in text.lower() for t in triggers):
        score += 1.5
        
    # Итоговая нормализация: скор не может быть меньше 1 и больше 5
    return min(5, max(1, int(round(score))))

# --- LLM ИНФЕРЕНС (С FALLBACK) ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt: str, schema_json: dict, client: OpenAI) -> SignalCard:
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты ведущий финансовый аналитик. Твоя задача — осуществлять строгую структуризацию текстовых сигналов в валидный JSON строго по заданной схеме."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return SignalCard.model_validate_json(response.choices[0].message.content)

@st.cache_data(show_spinner=False)
def process_with_llm_safe(article: Dict, api_key: str):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    all_sources = [article['url']] + article.get('extra_sources', [])
    schema_json = SignalCard.model_json_schema()
    prompt = f"Анализируемый текст: {article['text']}\nИсточники: {', '.join(all_sources)}\nСформируй ответ в строгом соответствии с JSON-схемой:\n{json.dumps(schema_json, ensure_ascii=False)}"
    
    try:
        result = call_llm(prompt, schema_json, client)
        final_hotness = calculate_hybrid_hotness(result.llm_hotness, article['text'], all_sources)
        return {"status": "success", "data": result, "final_hotness": final_hotness, "raw": article}
    
    except Exception as e:
        fallback_card = SignalCard(
            is_noise=False,
            headline=f"🚨 ИНЦИДЕНТ АНАЛИТИКИ: Требуется ручной разбор",
            llm_hotness=5,
            why_now="Модель не смогла структурировать данные в JSON. Необходимо прочитать оригинал.",
            category="ОШИБКА ПАРСИНГА",
            sources=all_sources,
            summary=f"Оригинальный текст не был обработан LLM. Системная ошибка: {str(e)[:150]}...",
            draft=f"Сырой текст для самостоятельного анализа:\n\n{article['text']}"
        )
        return {"status": "fallback", "data": fallback_card, "final_hotness": 5, "raw": article, "error_msg": str(e)}

# --- ВЫЧИСЛЕНИЕ МЕТРИК ---
def calculate_deduplication_metrics(threshold: float):
    tp = fp = fn = tn = 0
    for text1, text2, true_label in GOLDEN_DATASET:
        emb1 = embedding_model.encode([text1])[0]
        emb2 = embedding_model.encode([text2])[0]
        sim = cosine_similarity([emb1], [emb2])[0][0]
        pred_label = 1 if sim >= threshold else 0
        if pred_label == 1 and true_label == 1: tp += 1
        elif pred_label == 1 and true_label == 0: fp += 1
        elif pred_label == 0 and true_label == 1: fn += 1
        else: tn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
