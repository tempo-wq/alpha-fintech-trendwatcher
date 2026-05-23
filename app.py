import streamlit as st
import numpy as np
import pandas as pd

# --- ИМПОРТЫ ИЗ НАШИХ МОДУЛЕЙ ---
from src.config import RSS_SOURCES
from src.parser import fetch_rss_news
from src.ml_core import (
    deduplicate_articles_semantic, 
    process_with_llm_safe, 
    calculate_deduplication_metrics
)

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Fintech Trendwatcher MVP", page_icon="📈", layout="wide")
st.title("Fintech Trendwatcher MVP")
st.markdown("Модуль автоматизированного сбора, семантической дедупликации и экспертного анализа рыночных сигналов.")

# --- 2. САЙДБАР ---
st.sidebar.header("Конфигурация системы")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("Параметры математических моделей")
similarity_threshold = st.sidebar.slider(
    "Порог дедупликации (Cosine Similarity)", 
    min_value=0.50, max_value=0.99, value=0.70, step=0.01
)

if st.sidebar.button("Запустить аналитический пайплайн"):
    st.session_state.run_pipeline = True

# --- 3. ГЛАВНЫЙ ЦИКЛ ---
if st.session_state.get('run_pipeline', False):
    if not api_key:
        st.error("Ошибка аутентификации: отсутствует токен API.")
        st.stop()
        
    with st.spinner('Инициализация сбора данных по каналам RSS...'):
        raw_articles = fetch_rss_news(RSS_SOURCES, max_articles_per_feed=5)
    st.info(f"Общий объем сырых публикаций: {len(raw_articles)}")
    
    with st.spinner('Выполнение семантической дедупликации текстовых векторов...'):
        filtered_articles = deduplicate_articles_semantic(raw_articles, similarity_threshold)
    st.success(f"Уникальных сигналов после фильтрации: {len(filtered_articles)}")
    
    progress_bar = st.progress(0)
    final_digest = []
    quarantine = []  
    fallback_logs = [] 
    noise = []
    
    st.subheader("Мониторинг обработки потока нейросетевыми моделями")
    for i, article in enumerate(filtered_articles):
        res = process_with_llm_safe(article, api_key)
        
        if res["status"] == "success":
            card = res["data"]
            if card.is_noise:
                noise.append(card)
            else:
                final_digest.append((card, res["final_hotness"], res["raw"]))
        elif res["status"] == "fallback":
            final_digest.append((res["data"], res["final_hotness"], res["raw"]))
            fallback_logs.append(res)
        else:
            quarantine.append(res)
            
        progress_bar.progress((i + 1) / len(filtered_articles))
        
    # --- 4. UI: ТАБЫ ---
    tab1, tab2, tab3, tab4 = st.tabs(["Сформированный дайджест", "Информационный шум", "Журнал инцидентов (Логи)", "Контроль качества (Метрики)"])
    
    with tab1:
        final_digest.sort(key=lambda x: x[1], reverse=True)
        for card, hotness, raw_data in final_digest:
            is_fallback = "🚨" in card.headline
            hotness_indicators = "★" * hotness + "☆" * (5 - hotness)
            
            with st.expander(f"{'⚠️ [СПАСЕНО]' if is_fallback else ''} Приоритет: {hotness_indicators} | {card.category.upper()} | {card.headline}", expanded=is_fallback):
                st.markdown(f"**Актуальность и обоснование:** {card.why_now}")
                st.markdown(f"**Краткое содержание (Summary):** {card.summary}")
                
                if is_fallback:
                    st.error(f"**Оригинальные данные для разбора:**\n\n{card.draft}")
                else:
                    st.info(f"**Черновик аналитического материала:**\n\n{card.draft}")
                
                with st.expander("Технические параметры оценки (ML-Аналитика)"):
                    if not is_fallback:
                        st.write(f"**Декомпозиция скора:** Финальный индекс {hotness}/5 = (Базовый вес LLM * 0.4) + Модификаторы эвристик")
                    st.write(f"**Количество связанных дублирующих источников:** {len(card.sources)}")
                    if raw_data.get('merge_scores'):
                        st.caption(f"Сходство с дублями (Cosine Similarity): {raw_data['merge_scores']}")
                    st.caption(f"Проверяемые связанные URL-адреса: {', '.join(card.sources)}")
                
    with tab2:
        st.write(f"Отфильтровано нерелевантного контента: {len(noise)}")
        for n in noise:
            st.text(f"Исключено: {n.headline}")
            
    with tab3:
        st.write(f"Технические логи парсинга. Спасенных инцидентов (Fallback): {len(fallback_logs)}")
        for log in fallback_logs:
            st.warning(f"Ошибка LLM-пайплайна: {log['error_msg']}")
            st.code(log['raw']['url'])

    with tab4:
        st.subheader("Обоснование порога Cosine Similarity")
        st.write("Математический анализ компромисса точности и полноты (Precision-Recall Tradeoff) на контрольной выборке (Golden Dataset):")
        
        @st.cache_data(show_spinner=False)
        def generate_metrics_curve():
            threshold_grid = np.arange(0.50, 0.96, 0.01)
            records = []
            for t in threshold_grid:
                p, r, f = calculate_deduplication_metrics(t)
                records.append({
                    "Порог (Threshold)": round(t, 2),
                    "Precision": p,
                    "Recall": r,
                    "F1-Score": f
                })
            return pd.DataFrame(records).set_index("Порог (Threshold)")

        metrics_df = generate_metrics_curve()
        st.line_chart(metrics_df, height=300)
        
        prec, rec, f1 = calculate_deduplication_metrics(similarity_threshold)
        
        st.markdown(f"**Текущее рабочее состояние системы при пороге {similarity_threshold}:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (Точность)", f"{prec:.2f}")
        col2.metric("Recall (Полнота)", f"{rec:.2f}")
        col3.metric("F1-Score", f"{f1:.2f}")
        st.divider()
        
        st.subheader("Метрики отказоустойчивости (API Health & Fallback)")
        total_attempts = len(filtered_articles)
        
        success_count = len([x for x in final_digest if "🚨" not in x[0].headline]) + len(noise)
        fallback_count = len(fallback_logs)
        
        success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
        fallback_rate = (fallback_count / total_attempts * 100) if total_attempts > 0 else 0
        
        st.write(f"- Корректно структурировано LLM (Strict JSON): **{success_count}**")
        st.write(f"- Ошибка парсинга (Новость спасена эвристикой): **{fallback_count}**")
        
        col_api1, col_api2 = st.columns(2)
        col_api1.metric("Success Rate (Точность парсинга LLM)", f"{success_rate:.1f}%")
        col_api2.metric("Fallback Rate (Коэффициент деградации)", f"{fallback_rate:.1f}%")
        
        st.progress(success_rate / 100)