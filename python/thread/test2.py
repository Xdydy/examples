import asyncio


async def f(i):
    return i

async def main():
    promises = []
    result = []
    for i in range(10):
        task = asyncio.create_task(f(i))
        task.add_done_callback(lambda x: result.append(x.result()))
        promises.append(task)
    await asyncio.gather(*promises)
    return result


result = asyncio.run(main())
print(result)