# Files and Credentials Provided by Company

For your assignment, all the important information will be provided in the `.env` file attached here and the following resources and credentials so you can successfully build and test the AI Agent:

This repository expects (locally) `SHOPIFY_SHOP_NAME`, `SHOPIFY_ACCESS_TOKEN`, `SHOPIFY_API_VERSION` (use **2025-07**), and your own `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com).

---

### Shopify Admin API (2025-07)

- General Docs: https://shopify.dev/docs/api/admin-rest
- Orders Endpoint: https://shopify.dev/docs/api/admin-rest/2025-07/resources/order
- Products Endpoint: https://shopify.dev/docs/api/admin-rest/2025-07/resources/product
- Customers Endpoint: https://shopify.dev/docs/api/admin-rest/2025-07/resources/customer

### LangChain Docs

- ReAct Agent: https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.base.ReActChain.html
- Custom Tools: https://python.langchain.com/docs/how_to/custom_tools
- Python REPL Tool: https://python.langchain.com/api_reference/experimental/tools/langchain_experimental.tools.python.tool.PythonAstREPLTool.html

### Gemini API Docs

- https://ai.google.dev/gemini-api/docs

---

## 4. Questions Examples

Here is the set of **sample questions** that the agent should be able to handle, such as:

- “How many orders were placed in the last 7 days?”
- “Which products sold the most last month?”
- “Show a table of revenue by city.”
- “Who are my repeat customers?”
- “What is the AOV (Average Order Value) trend this month?”
- “Can you recommend what product to promote based on sales?”
- (Bonus) “Plot a graph of order volume over the past 4 weeks.”

---

## 5. Expectations and Constraints

- The agent must not generate or execute POST, PUT, or DELETE requests.
- All data fetching must go through your custom `get_shopify_data` tool.
- All data analysis must be performed using `PythonREPLAst`.
- Agent should **not emit raw code** in the response.
- Handle pagination, 429 rate limiting, and malformed API responses gracefully.

---

If you have any questions, feel free to ask before starting. Good luck!