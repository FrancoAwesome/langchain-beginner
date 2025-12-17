from fastmcp import FastMCP

mcp = FastMCP("francodemo")

@mcp.tool(description="Add two numbers together")
def add(a: int, b: int) -> int:
    """ Add two numbers together.
        Args:
            a (int): First number.
            b (int): Second number.
    """
    print(f"franco mcp demo called: add({a}, {b})")
    return a + b


@mcp.tool(description="Get weather of the city")
def weather(city: str) -> str:
    """ Return weather from city.
        Args:
            city (str): city name.
    """
    return "城市" + city + "，今天天气不错"


@mcp.resource("greeting://{name}")
def greet(name: str) -> str:
    """ Return greeting from name."""
    print(f"franco mcp demo called: greeting({name})")
    return name


if __name__ == "__main__":
    # sse start mode, using as a service
    # mcp.run(transport='sse')
    # stdio start mode, using as a method
    mcp.run(transport='stdio')
