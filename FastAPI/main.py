from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text : str = None
    isDone : bool = False

items= []


@app.get("/")
def root():
    return {"Hellow":"World"}


@app.post("/items")
def createItem(item:Item):
    items.append(item)
    return items

@app.get("/items")
def listItems(limit: int=10):
    return items[0:limit]

@app.get("/items/{item_id}")
def getItem(item_id: int) -> Item:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    