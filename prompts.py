from api_key import API_KEY
import asyncio
from mistralai import Mistral
import polars as pl
from datetime import datetime
import os
# Chemin vers le dossier data
DATA_DIR = "data"

model = "mistral-tiny" #because we need to pay for the API
mistral_client = Mistral(api_key=API_KEY)
output_file = "responses.parquet"

def import_data() -> pl.DataFrame:
    titles = pl.read_csv(
    source=os.path.join(DATA_DIR, "cryptopanic_news.csv")).select(pl.col("id").alias('newsId'), "title", "newsDatetime")
    currency = pl.read_csv(os.path.join(DATA_DIR, "currency.csv")).rename({'id':'currencyId'})
    ids = pl.read_csv(os.path.join(DATA_DIR, "news__currency.csv"))
    return (
        titles
        .join(ids, on="newsId")
        .join(currency, on='currencyId')
        .filter(pl.col("code").eq('BTC'))
        .with_columns(
            pl.col('newsDatetime').str.to_datetime().cast(pl.Date).alias('date')
        )
        .sort('newsDatetime')
        .select('date','title')
        .with_columns(
            prompts = pl.lit("""
    Here is a piece of news:
    "%s"
    Do you think this news will increase or decrease %s?
    Write your answer as: 
    {increase/decrease/uncertain}:
    {confidence (0-1)}:
    {magnitude of increase/decrease (0-1)}:
    {explanation (less than 25 words)}
            """)
        )
        .with_columns(pl.col('prompts').str.replace(r'%s', pl.col('title')))
        .with_columns(pl.col('prompts').str.replace(r'%s', "BTC"))
    )


async def stream_response(index: int, prompt: str, semaphore: asyncio.Semaphore,
                          retries: int = 4, backoff_base: float = 1) -> tuple:
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await mistral_client.chat.stream_async(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = ""
                async for chunk in response:
                    delta = chunk.data.choices[0].delta.content
                    if delta:
                        output += delta
                return index, prompt, output

            except Exception as e:
                err = str(e)
                # retry seulement si c'est un rate limit 429
                if "429" in err or "rate limit" in err.lower():
                    wait = backoff_base * (2 ** (attempt - 1))
                    print(f"[429] Rate limit, retry {attempt}/{retries} in {wait:.1f}s…")
                    await asyncio.sleep(wait)
                    continue
                # sinon on abandonne direct
                return index, prompt, f"[ERROR] {err}"

        # après tous les retries
        return index, prompt, "[ERROR] rate-limit retries exhausted"



async def process_in_batches(prompts: list, batch_size: int=1000, concurrency: int=50):
    semaphore = asyncio.Semaphore(concurrency)

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        print(f"Processing prompts {start} to {end - 1}...")
        tasks = [
            stream_response(i, prompt, semaphore)
            for i, prompt in enumerate(prompts[start:end], start=start)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])  # we want to maintain the order to map to dates

        yield results


def preprocess_responses(path_names = ['responses_2024_2025.csv', 'responses_2017_2024.csv']) -> pl.DataFrame:
    titles = import_data()
    responses = []
    for path in path_names:
        responses_frame = (
            pl.read_csv(path)
            .with_columns(
                pl.when(pl.col('response').str.slice(0,10).str.to_lowercase().str.contains('increase')).then(1).otherwise(0).alias('increase'),
                pl.when(pl.col('response').str.slice(0,10).str.to_lowercase().str.contains('decrease')).then(1).otherwise(0).alias('decrease'),
                pl.when(pl.col('response').str.slice(0,10).str.contains('[ERROR]')).then(1).otherwise(0).alias('error'),
            )
        )
        responses.append(responses_frame)
    responses = pl.concat(responses)
    responses = pl.concat([responses, titles], how='horizontal').select('date', 'title', 'increase', 'decrease', 'error')
    
    return (
        responses
        .select('date', 'increase', 'decrease')
        .with_columns(diff = pl.col('increase').sub(pl.col('decrease')))
        .group_by('date')
        .sum()
        .with_columns(score = pl.col('diff') / (pl.col('increase').add(pl.col('decrease'))))
        .sort('date')
    )

