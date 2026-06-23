import asyncio
import websockets

async def main():
    try:
        async with websockets.connect("ws://localhost:8000/asr") as ws:
            print("Connected successfully!")
            await asyncio.sleep(5)

    except Exception as e:
        print("Error:", e)

asyncio.run(main())