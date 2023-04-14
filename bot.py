import interactions
import sys
import signal
import os
import logging

import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity, get_embedding


# Load environment variables and API keys
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
discord_token = os.getenv('DISCORD_TOKEN')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read embeddings JSON file
embedding_json = pd.read_json('embeddings.json')

# Create Discord bot client
bot = interactions.Client(token=discord_token)

# Signal handler for clean exit


def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_prompt(df, user_input, mode='answer_question'):
    result, sources = search(df, user_input, n=3)

    if mode == 'answer_question':
        system_role = ("You are a Discord chatbot whose expertise is reading and summarizing a roleplaying game "
                       "rulebook called Mortal Reins. You are given a query and a series of text embeddings in order "
                       "of their cosine similarity to the query. You must take the given embeddings and return a "
                       "concise but accurate summary of the rules in the language of the query. Separate unrelated "
                       "rules by bullet points if it helps make the rules more clear. Provide examples if possible.")

        user_content = (f"Here is the question: {user_input}\n\n"
                        f"Here are the embeddings:\n\n"
                        f"1. {result.iloc[0]['text']}\n"
                        f"2. {result.iloc[1]['text']}\n"
                        f"3. {result.iloc[2]['text']}\n")

    else:
        system_role = ("You are a Discord chatbot whose expertise is reading a roleplaying game rulebook called "
                       "Mortal Reins and creating generative content based on it. You are given a query and a series "
                       "of text embeddings in order of their cosine similarity to the query. Generate creative "
                       "content based on the given embeddings that correspond to the query, using the themes and "
                       "information provided in the rulebook. Remember that Mortal Reins only uses d10 dice and "
                       "that the game is set in a custom fantasy world.")

        user_content = (f"Here is the base concept to generate from: {user_input}\n\n"
                        f"Here are the embeddings:\n\n"
                        f"1. {result.iloc[0]['text']}\n"
                        f"2. {result.iloc[1]['text']}\n"
                        f"3. {result.iloc[2]['text']}\n")

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content},
    ]

    logger.info('Done creating prompt')
    return messages, sources


def search(df, query, n=3):
    df = df.copy()  # prevent race condition
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(x, query_embedding))

    results = df.sort_values(
        "similarity", ascending=False, ignore_index=True).head(n)

    sources = [{'Page ' + str(results.iloc[i]['page']): results.iloc[i]
                ['text'][:150] + '...'} for i in range(n)]
    logger.info(sources)
    return results, sources


def gpt(messages, mode):
    temperature = 0.85 if mode == 'generate' else 0.7
    logger.info('Sending request to GPT-3')

    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=temperature, max_tokens=1500)

    answer = r.choices[0]["message"]["content"]
    logger.info('Done sending request to GPT-3')

    response = {'answer': answer}
    return response


def reply(df, question, mode='answer_question'):
    prompt, sources = create_prompt(df, question, mode)
    response = gpt(prompt, mode)
    response['sources'] = sources
    return response


@bot.command(name="question",
             description="Ask a question related to the Mortal Reins rulebook.",
             options=[
                 interactions.Option(
                     name="query",
                     description="Your question about the rulebook.",
                     type=interactions.OptionType.STRING,
                     required=True
                 )
             ])
async def question(ctx: interactions.CommandContext, query: str):
    logger.info(f"Received question: {query}")
    await ctx.defer()
    try:
        answer = reply(embedding_json, query)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await ctx.send(f"Sorry, I couldn't find an answer to your question due to an error: {str(e)}")
        return

    if 'answer' in answer:
        logger.info("Sending response to question")
        title = f"Answer to '{query}'"
        if len(title) > 250:
            title = title[:250] + "..."
        embed = interactions.Embed(
            title=title, description=answer['answer'], color=0x741420)

        # Add sources to the embed
        sources_text = ", ".join(
            [f"{k}" for source in answer['sources'] for k, v in source.items()])
        embed.add_field(
            name="For more information, you can check the following pages in the rulebook:", value=sources_text, inline=False)
        embed.set_footer(
            text="Please note that the answer provided may not be perfect. For the most accurate information, refer to the official rulebook.")

        await ctx.send(embeds=embed)
    else:
        await ctx.send("Sorry, I couldn't find an answer to your question.")


@bot.command(name="generate",
             description="Generate creative content based on the Mortal Reins rulebook.",
             options=[
                 interactions.Option(
                     name="query",
                     description="A theme or topic related to the rulebook.",
                     type=interactions.OptionType.STRING,
                     required=True
                 )
             ])
async def generate(ctx: interactions.CommandContext, query: str):
    logger.info(f"Received generate request: {query}")
    await ctx.defer()
    try:
        answer = reply(embedding_json, query, mode='generate')
    except Exception as e:
        logger.error(f"Error processing generate request: {str(e)}")
        await ctx.send(f"Sorry, I couldn't generate content based on your query due to an error: {str(e)}")
        return

    if 'answer' in answer:
        logger.info("Sending generated content")
        title = f"Generated content for '{query}'"
        if len(title) > 250:
            title = title[:250] + "..."
        embed = interactions.Embed(
            title=title, description=answer['answer'], color=0x741420)
        await ctx.send(embeds=embed)
    else:
        await ctx.send("Sorry, I couldn't generate content based on your query.")


@bot.command(name="help",
             description="Show help information for the Mortal Reins bot.")
async def help(ctx):
    await ctx.defer()
    embed = interactions.Embed(
        title="Mortal Reins Discord Bot Help",
        description="This bot specializes in reading and summarizing the Mortal Reins roleplaying game rulebook. It can also generate creative content based on the rulebook.",
        color=0x741420
    )
    embed.add_field(
        name="/question [query]",
        value="Ask the bot a question related to the Mortal Reins rulebook, and it will provide a summary of the most relevant rules.",
        inline=False
    )
    embed.add_field(
        name="/generate [query]",
        value="Provide a theme or topic related to the Mortal Reins rulebook, and the bot will generate creative content based on the most similar rulebook page.",
        inline=False
    )
    embed.add_field(
        name="/help",
        value="Shows this help message.",
        inline=False
    )
    embed.set_footer(
        text="Please note that the answers provided may not be perfect. For the most accurate information, refer to the official rulebook.")
    await ctx.send(embeds=embed)

logger.info('Starting bot')
bot.start()
