# Contains system_prompt and generate_input_prompt

system_prompt = (
    "You are an assistant that fetches housing listings using a provided tool. "
    "When given a query, decide whether to call the tool and return the JSON output."
)


def generate_input_prompt(song_or_query, artist=None):
    """
    Example placeholder for compatibility. For housing, pass the query string directly.
    """
    return song_or_query  # For housing, just echo the natural-language query
