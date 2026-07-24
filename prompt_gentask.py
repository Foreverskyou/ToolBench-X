prompt_markdown_sequential='''# Task: Generate Tool-Calling Evaluation Data

Your task is to generate data for evaluating an agent's tool-calling capabilities. You will create specific tool-calling tasks based on a main topic and its subtopics. Each task must require **sequential, dependent tool calls** (i.e., Tool A must be completed before Tool B can run).

**Important constraint:** All tools must **not involve image processing, image generation, or any visual content**. Tools should test text-based operations.

---

## Main Topic
**Technology & Engineering**

## Subtopics
- Device setup and configuration

---

## Requirements

### General Requirements
- You MUST generate **exactly five distinct tasks**.
- Each task must be solvable **only through a sequence of tool calls**.
- Each task must include **explicit dependencies between tools**  
  (i.e., outputs from earlier tools are required by later tools).
- Tools must be **relevant to the topic and subtopics**.

### User Prompt Constraints
- `user_prompt` must **explicitly instruct the model what to answer**.  
- `user_prompt` **must not contain multiple questions**; the agent should produce **a single final answer**.
- It **MUST NOT**:
  - Reveal execution steps  
  - Suggest intermediate procedures  
  - Indicate tool usage or order  

### Answer Constraints
- Each task must produce a final answer that is:
  - **Programmatically verifiable**, OR  
  - A **single, objectively correct value**
- `final_answer` must be:
  - **Concise**
  - **Result-only** (no explanation, no steps)

### Diversity Requirement
- The five tasks must be **diverse in scenario and complexity**.

---

## Output Format (STRICT JSON)

You MUST output a valid JSON object with the following schema:

```json
{
  "tasks": [
    {
      "id": "task_1",
      "user_prompt": "string",
      "tools_used": [
        "tool_name_1",
        "tool_name_2",
        "tool_name_3",
        "tool_name_4",
      ],
      "final_answer": "string"
    }
  ]
}'''

prompt_markdown_parallel='''# Task: Generate Parallel Tool-Calling Evaluation Data

Your task is to generate data for evaluating an agent's **parallel tool-calling capabilities**. You will create specific tasks based on a main topic and its subtopics. Each task must require **all tools to be used**, but **no tool may depend on the output of another** (true parallel execution).

**Important constraint:** All tools must **not involve image processing, image generation, or any visual content**. Tools should test text-based operations.

---

## Main Topic
**Technology & Engineering**

## Subtopics
- Device setup and configuration

---

## Requirements

### General Requirements
- You MUST generate **exactly five distinct tasks**.
- Each task must require **all listed tools to be used**.
- Tools must be **independent**:
  - No tool's output may be used as input to another tool.
- Tools must be **relevant to the topic and subtopics**.

### User Prompt Constraints
- `user_prompt` must **explicitly instruct the model what to answer**.  
- `user_prompt` **must not contain multiple questions**; the agent should produce **a single final answer**.
- It **MUST NOT**:
  - Reveal execution steps  
  - Suggest intermediate procedures  
  - Indicate tool usage or order  

### Answer Constraints
- Each task must produce a final answer that is:
  - **Programmatically verifiable**, OR  
  - A **single, objectively correct value**
- `final_answer` must be:
  - **Concise**
  - **Result-only** (no explanation, no steps)

### Diversity Requirement
- The five tasks must be **diverse in scenario and complexity**.

---

## Output Format (STRICT JSON)

You MUST output a valid JSON object with the following schema:

```json
{
  "tasks": [
    {
      "id": "task_1",
      "user_prompt": "string",
      "tools_used": [
        "tool_name_1",
        "tool_name_2",
        "tool_name_3",
        "tool_name_4"
      ],
      "final_answer": "string"
    }
  ]
}'''

prompt_markdown_mixture='''# Task: Generate Mixed Tool-Calling Evaluation Data

Your task is to generate data for evaluating an agent's **mixed tool-calling capabilities**. Each task must include **both sequential (dependent) and parallel (independent) tool calls**. All tools must be used to complete the task.

**Important constraint:** All tools must **not involve image processing, image generation, or any visual content**. Tools should test text-based operations.

---

## Main Topic
**Technology & Engineering**

## Subtopics
- Device setup and configuration

---

## Requirements

### General Requirements
- You MUST generate **exactly five distinct tasks**.
- Each task must include **all listed tools**.
- Tools may be a mix of:
  - **Sequential**: some tools must be called in order (output of one used as input to the next)  
  - **Parallel**: some tools are independent and can be called in any order
- Tools must be **relevant to the topic and subtopics**.

### User Prompt Constraints
- `user_prompt` must **explicitly instruct the model what to answer**.  
- `user_prompt` **must not contain multiple questions**; the agent should produce **a single final answer**.
- It **MUST NOT**:
  - Reveal execution steps  
  - Suggest intermediate procedures  
  - Indicate tool usage or order  

### Answer Constraints
- Each task must produce a final answer that is:
  - **Programmatically verifiable**, OR  
  - A **single, objectively correct value**
- `final_answer` must be:
  - **Concise**
  - **Result-only** (no explanation, no steps)

### Diversity Requirement
- The five tasks must be **diverse in scenario and complexity**.

---

## Output Format (STRICT JSON)

You MUST output a valid JSON object with the following schema:

```json
{
  "tasks": [
    {
      "id": "task_1",
      "user_prompt": "string",
      "tools_used": [
        {
          "tool_name": "string",
          "type": "sequential" | "parallel"
        }
      ],
      "final_answer": "string"
    }
  ]
}'''