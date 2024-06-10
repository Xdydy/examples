import asyncio

async def f():
    return 1

def main():
    loop = asyncio.get_event_loop()
    task = loop.create_task(f())
    return task.result()

print(main())