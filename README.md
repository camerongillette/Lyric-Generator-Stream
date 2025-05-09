## Intro
This is the backend of a lyric generator that will allow you to stream lyrics gradually as the model creates them. 
This is super useful as we can easily interupt the ml as needed so it doesn't just output the whole song.
And when paired with the frontend() we are able to an app that's far more like a shared google doc instead of an awkward back and forth.

## Runnning 
To start the backend on your local
uvicorn ragLyricGenerator:app --reload


## Requirements

We'll need to make sure we have our OpenAI token id as an environment variable
We'll also need to make an .env file that has FRONTEND_URL defined. For example

FRONTEND_URL=http://localhost:5173

You'll also need a few python libraries that you'll need to install with pip. It should give you the exact commands to run

You'll need to install weaviate to store all the embeddings. It's super overkill to use this as it calculates the embeddings so quickly
