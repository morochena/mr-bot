import os

import discord
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity, get_embedding

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def create_prompt(df, user_input, mode='summarize'):
    if mode == 'summarize':
        result = search(df, user_input, n=3)
        system_role = "You are a Discord chatbot whose expertise is reading and summarizing a roleplaying game rulebook called Mortal Reins. You are given a query, a series of text embeddings in order of their cosine similarity to the query. You must take the given embeddings and return a concise but accurate summary of the rules in the language of the query. Separate unrelated rules by bullet points if it helps make the rules more clear. Provide examples if possible."
    else:
        result = search(df, user_input, n=1)
        system_role = "You are a Discord chatbot whose expertise is reading a roleplaying game rulebook called Mortal Reins and generating content based on it. You are given a query and a text embedding that is most similar to the query. Generate creative content based on the given embedding, maintaining the theme and style of the rulebook."

    user_content = """Here is the question: """ + user_input + """
            
Here are the embeddings: 
            
1.""" + str(result.iloc[0]['text']) + ("""
2.""" + str(result.iloc[1]['text']) + """
3.""" + str(result.iloc[2]['text']) + """\n""" if mode == 'summarize' else "\n")

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content},]

    print('Done creating prompt')
    return messages


def search(df, query, n=3, pprint=True):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(x, query_embedding))

    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    results = results.head(n)
    global sources
    sources = []
    for i in range(n):
        sources.append(
            {'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
    print(sources)
    return results.head(n)


def gpt(messages):
    print('Sending request to GPT-3')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, max_tokens=1500)
    answer = r.choices[0]["message"]["content"]
    print('Done sending request to GPT-3')
    response = {'answer': answer, 'sources': sources}
    return response


def reply(df, question, mode='summarize'):
    prompt = create_prompt(df, question, mode)
    response = gpt(prompt)
    return response


embedding_json = pd.read_json('embeddings.json')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

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
