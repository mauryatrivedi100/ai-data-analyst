"""AI insights module for generating analysis using a local Ollama LLM (Llama 3.2)."""

import re
import requests


# Ollama API endpoint (default local server)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 120  # seconds — local models can be slower


def build_prompt(context):
    """Construct a structured prompt with dataset context for the LLM.

    Args:
        context: A dict with keys: dataset_summary, statistics, correlations, model_results

    Returns:
        A formatted prompt string.
    """
    prompt_parts = [
        "You are an expert data analyst. Analyze the following dataset information and provide comprehensive insights.",
        "",
        "Please structure your response with the following sections:",
        "## Overview",
        "Provide a high-level summary of the dataset and its characteristics.",
        "",
        "## Key Observations",
        "List the most important patterns, trends, and statistical findings.",
        "",
        "## Business Insights",
        "Provide actionable business intelligence derived from the data.",
        "",
        "## Potential Risks",
        "Identify data quality issues, biases, or limitations that could affect analysis.",
        "",
        "## Recommendations",
        "Provide specific, actionable recommendations based on the analysis.",
        "",
        "---",
        "",
        "Here is the dataset information to analyze:",
        "",
    ]

    if context.get("dataset_summary"):
        prompt_parts.append("### Dataset Summary")
        prompt_parts.append(str(context["dataset_summary"]))
        prompt_parts.append("")

    if context.get("statistics"):
        prompt_parts.append("### Descriptive Statistics")
        prompt_parts.append(str(context["statistics"]))
        prompt_parts.append("")

    if context.get("correlations"):
        prompt_parts.append("### Correlation Information")
        prompt_parts.append(str(context["correlations"]))
        prompt_parts.append("")

    if context.get("model_results"):
        prompt_parts.append("### Model Results")
        prompt_parts.append(str(context["model_results"]))
        prompt_parts.append("")

    return "\n".join(prompt_parts)


def call_ollama_api(prompt):
    """Send a request to the local Ollama API.

    Args:
        prompt: The prompt string to send.

    Returns:
        The raw text response from the LLM.

    Raises:
        TimeoutError: If Ollama does not respond within the timeout.
        RuntimeError: If Ollama returns an error or is unreachable.
        ConnectionError: If Ollama server is not running.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "AI service did not respond in time. "
            "Ensure Ollama is running and the model is loaded."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama. "
            "Please ensure Ollama is running (ollama serve) on localhost:11434."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"AI service error: {e}")

    if response.status_code != 200:
        error_detail = response.text[:200] if response.text else "Unknown error"
        raise RuntimeError(
            f"Ollama returned status {response.status_code}: {error_detail}"
        )

    data = response.json()
    return data.get("response", "")


def parse_insights_response(response):
    """Parse the LLM response into structured sections.

    Args:
        response: The raw text response from the LLM.

    Returns:
        A dict with keys: overview, observations, business_insights, risks, recommendations.
    """
    sections = {
        "overview": "",
        "observations": "",
        "business_insights": "",
        "risks": "",
        "recommendations": "",
    }

    if not response:
        return sections

    # Define section header patterns and their corresponding keys
    section_patterns = [
        (r"##?\s*(?:Dataset\s+)?Overview", "overview"),
        (r"##?\s*Key\s+Observations?", "observations"),
        (r"##?\s*Business\s+Insights?", "business_insights"),
        (r"##?\s*(?:Potential\s+)?Risks?", "risks"),
        (r"##?\s*(?:Actionable\s+)?Recommendations?", "recommendations"),
    ]

    # Find all section positions
    section_positions = []
    for pattern, key in section_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            section_positions.append((match.start(), match.end(), key))

    # Sort by position in text
    section_positions.sort(key=lambda x: x[0])

    if not section_positions:
        # If no sections found, put everything in overview
        sections["overview"] = response.strip()
        return sections

    # Extract content for each section
    for i, (start, end, key) in enumerate(section_positions):
        if i + 1 < len(section_positions):
            next_start = section_positions[i + 1][0]
            content = response[end:next_start]
        else:
            content = response[end:]

        sections[key] = content.strip()

    return sections


def generate_insights(dataset_summary, statistics, correlations, model_results):
    """Orchestrate the full insights generation pipeline using local Ollama LLM.

    Args:
        dataset_summary: Summary information about the dataset.
        statistics: Descriptive statistics for the dataset.
        correlations: Correlation information between columns.
        model_results: ML model training results (can be None).

    Returns:
        A dict with keys: overview, observations, business_insights, risks, recommendations.

    Raises:
        TimeoutError: If Ollama does not respond within the timeout.
        RuntimeError: If Ollama returns an error.
        ConnectionError: If Ollama is not running.
    """
    context = {
        "dataset_summary": dataset_summary,
        "statistics": statistics,
        "correlations": correlations,
        "model_results": model_results,
    }

    prompt = build_prompt(context)

    try:
        raw_response = call_ollama_api(prompt)
    except TimeoutError:
        raise TimeoutError("AI insights generation timed out. Please try again.")
    except ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama. Please ensure it is running."
        )
    except Exception as e:
        raise RuntimeError(f"AI insights generation failed: {e}")

    insights = parse_insights_response(raw_response)
    return insights
