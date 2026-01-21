import asyncio

from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"sync -> a: {a}, b: {b}")
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"async -> a: {a}, b: {b}")
    return a * b

async def main():
    # func --> sync func, invoke func in sync context
    # coroutine --> async func, invoke func in async context
    calculator = StructuredTool.from_function(
        func=multiply,
        coroutine=amultiply,
        return_result=True
    )
    # sync -> a:1, b:2
    print(calculator.invoke({"a":1,"b":2}))
    # async -> a:2, b:5
    print(await calculator.ainvoke({"a":2,"b":5}))

asyncio.run(main())