import asyncio
from api.database import connect_to_mongo, get_db

async def run():
    await connect_to_mongo()
    db = get_db()
    messages = await db["messages"].find().to_list(length=10)
    print(f"Found {len(messages)} messages total in db.")
    for m in messages:
        print(f"Session: {m['session_id']}, Role: {m['role']}, text: {m['content'][:30]}")

asyncio.run(run())
