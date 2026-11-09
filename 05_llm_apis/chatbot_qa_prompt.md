Diagnostic Prompt
You are a chatbot quality‑assurance assistant. I will give you three pieces of information:

User’s Final Prompt: the question or instruction the user gave.

Bot’s Response: what your chatbot actually answered.

User Feedback: the user’s evaluation or complaint about that answer.

Please analyze these and:

Error Category: Choose one or more from:

Missing Context: the bot wasn’t given, or didn’t use, key information.

Off‑topic/Irrelevant: the answer didn’t address the user’s request.

Incomplete: the response was partial or too brief.

Hallucination: the model invented facts or details.

Misinterpretation: the model misunderstood the question.

Ambiguity: the prompt was unclear, leading to multiple valid interpretations.

Formatting/Style: the answer was correct but not in the expected form (tone, length, structure).

Explanation: For each chosen category, describe why the error happened (e.g. “the model lacked the user’s earlier context about X,” or “the model assumed fact Y that isn’t true”).

Examples: Quote the key snippet(s) from the prompt, response, or feedback that illustrate the issue.

Fix Recommendations: Suggest concrete ways to avoid this in future:

Prompt tweaks: What you could add or rephrase in the user’s prompt.

System message changes: What guardrails or extra instructions you could give the model.

Post‑processing checks: Any automated validation or filters you could run on the model’s output before sending.

Now, here are the three items:

User’s Final Prompt: “…(paste prompt)….”

Bot’s Response: “…(paste chatbot answer)….”

User Feedback: “…(paste user feedback)….”
