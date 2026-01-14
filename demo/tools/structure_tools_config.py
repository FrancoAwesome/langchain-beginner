import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculateInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


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
        name="multiply",
        description="multiply two numbers",
        args=CalculateInput,
        return_direct=True
    )
    # sync -> a:1, b:2
    print(calculator.invoke({"a":1,"b":2}))
    # async -> a:2, b:5
    print(await calculator.ainvoke({"a":2,"b":5}))

asyncio.run(main())