import asyncio

queue = asyncio.Queue()

async def func():
    async def f():
        print("F Start")
        await asyncio.sleep(3)
        print("finish")
        await queue.put(1)
        return 2
    task = asyncio.create_task(f())
    return task


async def main():
    await func()
    # 等待task完成
    await asyncio.sleep(1)
    print("waiting")
    res = await queue.get()
    print(res)
    # result = await task
    # print(result)
   
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()