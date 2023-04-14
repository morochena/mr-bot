import sys
import signal
import os

import discord
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity, get_embedding

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

embedding_json = pd.read_json('embeddings.json')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_prompt(df, user_input, mode='summarize'):
    if mode == 'summarize':
        result, sources = search(df, user_input, n=3)
        system_role = "You are a Discord chatbot whose expertise is reading and summarizing a roleplaying game rulebook called Mortal Reins. You are given a query and a series of text embeddings in order of their cosine similarity to the query. You must take the given embeddings and return a concise but accurate summary of the rules in the language of the query. Separate unrelated rules by bullet points if it helps make the rules more clear. Provide examples if possible."
    else:
        result, sources = search(df, user_input, n=3)
        system_role = "You are a Discord chatbot whose expertise is reading a roleplaying game rulebook called Mortal Reins and creating generative content based on it. You are given a query and a series of text embeddings in order of their cosine similarity to the query. Generate creative content based on the given embeddings that correspond to the query, using the themes and information provided in the rulebook. Remember that Mortal Reins only uses d10 dice and that the game is set in a custom fantasy world."

    user_content = """Here is the question: """ + user_input + """
            
Here are the embeddings: 
            
1.""" + str(result.iloc[0]['text']) + ("""
2.""" + str(result.iloc[1]['text']) + """
3.""" + str(result.iloc[2]['text']) + """\n""" if mode == 'summarize' else "\n")

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content},]

    print('Done creating prompt')
    return messages, sources


def search(df, query, n=3):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(x, query_embedding))

    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    results = results.head(n)

    sources = []
    for i in range(n):
        sources.append(
            {'Page ' + str(results.iloc[i]['page']): results.iloc[i]['text'][:150] + '...'})
    print(sources)
    return results.head(n), sources


def gpt(messages, mode):
    temperature = 0.7
    if mode == 'generate':
        temperature = 0.85
    print('Sending request to GPT-3')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=temperature, max_tokens=1500)
    answer = r.choices[0]["message"]["content"]
    print('Done sending request to GPT-3')

    response = {'answer': answer}
    return response


def reply(df, question, mode='summarize'):
    prompt, sources = create_prompt(df, question, mode)
    response = gpt(prompt, mode)
    response['sources'] = sources
    return response


@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!help'):
        embed = discord.Embed(
            title="Mortal Reins Discord Bot Help",
            description="This bot specializes in reading and summarizing the Mortal Reins roleplaying game rulebook. It can also generate creative content based on the rulebook.",
            color=0x741420
        )
        embed.add_field(
            name="!question [query]",
            value="Ask the bot a question related to the Mortal Reins rulebook, and it will provide a summary of the most relevant rules.",
            inline=False
        )
        embed.add_field(
            name="!generate [query]",
            value="Provide a theme or topic related to the Mortal Reins rulebook, and the bot will generate creative content based on the most similar rulebook page.",
            inline=False
        )
        embed.add_field(
            name="!help",
            value="Shows this help message.",
            inline=False
        )
        embed.set_footer(
            text="Please note that the answers provided may not be perfect. For the most accurate information, refer to the official rulebook.")
        await message.channel.send(embed=embed)

    if message.content.startswith('!question'):
        query = message.content[10:]
        try:
            answer = reply(embedding_json, query)
        except Exception as e:
            await message.channel.send(f"Sorry, I couldn't find an answer to your question due to an error: {str(e)}")
            return

        if 'answer' in answer:
            title = f"Answer to '{query}'"
            if len(title) > 250:
                title = title[:250] + "..."
            embed = discord.Embed(
                title=title, description=answer['answer'], color=0x741420)

            # Add sources to the embed
            sources_text = ", ".join(
                [f"{k}" for source in answer['sources'] for k, v in source.items()])
            embed.add_field(
                name="For more information, you can check the following pages in the rulebook:", value=sources_text, inline=False)
            embed.set_footer(
                text="Please note that the answer provided may not be perfect. For the most accurate information, refer to the official rulebook.")

            await message.channel.send(embed=embed)
        else:
            await message.channel.send("Sorry, I couldn't find an answer to your question.")

    elif message.content.startswith('!generate'):
        query = message.content[9:]
        try:
            answer = reply(embedding_json, query, mode='generate')
        except Exception as e:
            await message.channel.send(f"Sorry, I couldn't generate content based on your query due to an error: {str(e)}")
            return

        if 'answer' in answer:
            title = f"Generated content for '{query}'"
            if len(title) > 250:
                title = title[:250] + "..."
            embed = discord.Embed(
                title=title, description=answer['answer'], color=0x741420)
            await message.channel.send(embed=embed)
        else:
            await message.channel.send("Sorry, I couldn't generate content based on your query.")


client.run(os.getenv('DISCORD_TOKEN'))

# interative cli if you pass in the --cli flag
# otherwise run the bot
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cli', action='store_true')
#     args = parser.parse_args()

#     if args.cli:
#         while True:
#             query = input('Enter your question: ')
#             try:
#                 answer = reply(embedding_json, query)
#             except Exception as e:
#                 print(
#                     f"Sorry, I couldn't find an answer to your question due to an error: {str(e)}")
#                 continue

#             if 'answer' in answer:
#                 print(f"Answer to '{query}': {answer['answer']}")
#             else:
#                 print("Sorry, I couldn't find an answer to your question.")
#     else:
#         client.run(os.getenv('DISCORD_TOKEN'))
