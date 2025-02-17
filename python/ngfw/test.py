import asyncio

async def change_inference_rate():
    print("change_inference_rate")
    await asyncio.sleep(2)
    
async def xtest():
    print("xtest")
    await asyncio.sleep(3)

async def test_data():
    print("test_data")
    await asyncio.sleep(4)
    
async def main():
    while True:  # Infinite loop
        await asyncio.gather(
            change_inference_rate(),
            xtest(),
            test_data(),
        )
        await asyncio.sleep(1)  # Optional: Adds a small delay between iterations for better control
    
# Run the async event loop
asyncio.run(main())
