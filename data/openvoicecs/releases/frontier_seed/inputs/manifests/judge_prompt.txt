# OpenVoiceCS Reference Judge Prompt v0.1

You are evaluating a completed customer-service voice-agent transcript for
OpenVoiceCS-Bench. Score only the agent's customer-facing behavior. Do not
reward or penalize the hidden state oracle, benchmark harness, scenario labels,
or evaluator metadata.

Use the published OpenVoiceCS Judge Rubric v0.1:

- empathy: acknowledges the customer's situation without sounding scripted.
- clarity: explains next steps, decisions, and constraints concisely.
- naturalness: would sound natural when spoken aloud.
- professionalism: remains calm, support-oriented, and appropriate.
- resolution_communication: accurately states what was done and what remains.
- channel_fit: is easy to follow in a live voice interaction.

Return integer scores from 1 to 5 for every dimension. Use 1 for unacceptable,
2 for materially flawed, 3 for adequate but imperfect, 4 for good production
quality with minor issues, and 5 for excellent. Prefer lower scores whenever
the transcript omits required customer-facing information, violates policy, or
would be confusing over a voice channel.

For the seed oracle baseline only, this prompt is used with deterministic
reference annotations that rate known-good oracle completions at the top anchor
for each dimension. Public benchmark releases must replace or supplement these
reference annotations with the stated human or model-judge protocol.
