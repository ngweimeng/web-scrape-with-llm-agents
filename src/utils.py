# Supplies default_values for retry error callback

def default_values(retry_state):
    """
    Return a default empty response when retries fail.
    """
    return {"error": "Request failed after retries"}
