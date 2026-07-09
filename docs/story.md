# The Smart Refinery Planning Team (non-math narrative)

Imagine your refinery must decide every day which crudes to buy, how to run the units, and what to blend — without solving one giant 10,000-piece puzzle.

**The Boss (Master / ADMM loop)** sets price signals for the connecting streams (naphtha, diesel precursors, etc.) and asks each department for its best plan.

**Department experts (sub-agents)** each own one area:

- CDU — yields and charge limits
- Tank Farm — inventory and timing
- Blender — product specs and recipes
- Utilities — steam, power, fuel gas

Each expert is a **math solver + LLM brain**: the solver enforces hard rules; the LLM notes nonlinear yields, soft business rules, and smart what-ifs.

After a few quick rounds of prices down / proposals up, the team converges on a plan with **shadow prices** (marginal values) that planners can trust for make-buy-sell decisions — same economic language as PIMS, faster and more parallel.
