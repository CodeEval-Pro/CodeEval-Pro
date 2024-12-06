PROMPT_WRAPPER = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}

@@ Response
{response}"""

COT_PROMPT_WRAPPER = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}

@@ Response
Let's think step by step.
{response}"""

PROMPT_WRAPPER_ONE_SHOT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}

@@ Response
Let's think step by step.
{response}"""

PROMPT_WRAPPER_API = """
@@ Instruction
{instruction}
@@ Response
"""