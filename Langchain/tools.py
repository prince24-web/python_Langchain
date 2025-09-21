from langchain_core.tools import tool

@tool
def multipy(a: int, b : int) -> int:
    """Multiply 2 numbers"""
    return a * b

#using tool directly
print(multipy.invoke({"a":3, "b":5}))

#inspecting tool schema
print(multipy.name)
print(multipy.description)
print(multipy.args)