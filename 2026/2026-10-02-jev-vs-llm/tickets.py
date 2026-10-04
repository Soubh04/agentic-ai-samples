"""Sample support tickets. All fictional: no real customers, products or companies."""

SAMPLE_TICKETS = [
    # Clear billing, with a deadline and a refund request.
    "I was charged twice for my March invoice. Please refund the duplicate payment today.",
    # Clear technical, and the customer is blocked.
    "Login fails with a 500 error since this morning and our whole team is blocked. Please help ASAP.",
    # Clear sales.
    "We are 40 people and would like a demo of the enterprise plan, plus a quote for annual pricing.",
    # Two teams could own it: a good case for a person, or a model that can reason.
    "My invoice shows an error and I cannot log in to download it. Not sure who handles this.",
    # Too vague to route.
    "Hello, can someone get back to me about my account when possible? Thanks.",
    # HTML entities and an emoji: sent to the backends exactly as written.
    "I was billed &pound;94 instead of &pound;49 &amp; I'd like a refund of the difference \U0001F4B8. Invoice #1042 attached.",
]
