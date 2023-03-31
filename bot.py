import os

import discord
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity, get_embedding

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def create_prompt(df, user_input):
    result = search(df, user_input, n=3)
    system_role = """whose expertise is reading and summarizing a roleplaying game rulebook. You are given a query, 
        a series of text embeddings in order of their cosine similarity to the query. 
        You must take the given embeddings and return a detailed summary of the rules in the languange of the query: 
            
        Here is the question: """ + user_input + """
            
        and here are the embeddings: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """
        """

    user_content = f"""Given the question: "{str(user_input)}". Return a detailed answer based on the rulebook:"""

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
    # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
    results = results.head(n)
    global sources
    sources = []
    for i in range(n):
        # append the page number and the text as a dict to the sources list
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


def reply(df, question):
    prompt = create_prompt(df, question)
    response = gpt(prompt)
    return response


# read embedding.json file
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
        answer = reply(embedding_json, query)

        if 'answer' in answer:
            embed = discord.Embed(
                title=f"Answer to '{query}'", description=answer['answer'], color=discord.Color.green())
            await message.channel.send(embed=embed)

client.run(os.getenv('DISCORD_TOKEN'))
