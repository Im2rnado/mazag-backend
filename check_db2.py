import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test_mongo():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.mazag
    count = await db.messages.count_documents({})
    msgs = await db.messages.find().sort("created_at", -1).limit(5).to_list(5)
    print(f"Count: {count}")
    for m in msgs:
        print(f"Role: {m.get('role')} SID: {m.get('session_id')} Text: {m.get('content')[:20]}")

asyncio.run(test_mongo())
