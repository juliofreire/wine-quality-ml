from fastapi import FastAPI

# Instantiate the app.
app = FastAPI()

# Define GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}