# AI Metacognition Project

A longitudinal 21-day experiment platform that embeds an OpenAI LLM conversation
inside a Qualtrics survey. Participants complete a **baseline survey on Day 1**,
then a **daily follow-up survey** for 21 days. Each daily session includes a
topic-locked LLM conversation and metacognitive outcome measures.

---

## Measurement Scales

Three instruments are used: one administered once at baseline, two repeated after each daily LLM session. The baseline measure captures stable individual differences in how people think about thinking, while the daily measures track how those experiences shift over the course of the study.

| Scale | Type | When | Source |
|---|---|---|---|
| MCQ-30 | Trait metacognition | Baseline only | Wells & Cartwright-Hatton (2004) |
| Processing Fluency | State metacognition | Daily, post-chat | Shulman & Sweitzer (2018a, 2018b) |
| Perceived Knowledge | State confidence | Daily, post-chat | Shulman & Sweitzer (2018b) |

**MCQ-30** measures stable beliefs about thinking and worrying across five subscales (positive beliefs about worry, uncontrollability and danger, cognitive confidence, need to control thoughts, cognitive self-consciousness). Given once at baseline to account for individual differences. Subscales like cognitive confidence and cognitive self-consciousness are particularly relevant here, since they speak to how people monitor and evaluate their own understanding, which is likely to interact with how they engage with LLM-generated information over time.

**Processing Fluency** (3 items) asks how smoothly participants felt they processed the LLM's responses. Administered after each chat session, it serves as the primary within-person metacognition measure across the 21 days. Items:

1. Overall, I felt that the language used in these questions was difficult. *(reverse-coded)*
2. The information presented in these questions felt new to me. *(reverse-coded)*
3. It was easy for me to provide my opinions when answering these questions.

**Perceived Knowledge** (adapted from Shulman & Sweitzer's Perceived Political Knowledge measure) asks how informed and confident participants feel about the session topic. It covers both on-topic knowledge (directly tied to what came up in the conversation) and off-topic knowledge (broader domain awareness). This is the second daily outcome variable and is used alongside processing fluency in the longitudinal analyses and knowledge network correlations. Items are adapted for each condition (`poli_conspiracy` / `selective_exposure`).

---

