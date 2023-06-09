# Mortal Reins QA Bot

This is a fan-made Discord bot for [Mortal Reins](https://www.mythicvisionsgames.com/). It lets you make queries against the rulebook. 

![Sample Usage](./sample.png)


## Requirements
- python 3.10.6
- Discord API token
- OpenAI API key
- Mortal Reins PDF

## Installation
- Clone the repository
- Install the requirements with `pip install -r requirements.txt`
- Download Mortal Reins PDF to `book.pdf`
- Set environment variables in `.env`
- Generate the book with `python generate_embeddings.py`
- Run Discord bot with `python bot.py`


Example `.env` file:
```
OPENAI_API_KEY=sk-...
DISCORD_TOKEN=MTA...
```
