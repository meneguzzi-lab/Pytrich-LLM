#!/usr/bin/env python3
"""
Generic Heuristic Generator for HDDL Planning

This script generates domain-dependent heuristics using LLMs.
Supports Google Gemini, OpenAI, and Anthropic APIs.

Modes:
  --batch N      Generate N independent heuristics (one-shot, no feedback)
  --count N      Iterative refinement: generate v0, test, refine to v1, ..., vN-1
  --output-prompt FILE  Export complete prompt to file for use in any AI tool

Usage:
    python generate_heuristic.py --domain towers --model gemini25pro --batch 20
    python generate_heuristic.py --domain towers --model claudesonnet45 --count 3
    python generate_heuristic.py --domain towers --model gemini25pro --output-prompt /tmp/prompt.txt
    python generate_heuristic.py --domain towers --model gemini25pro --dry-run
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
import yaml

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from templates.prompt_template import DEFAULT_COMPONENTS, generate_prompt, generate_refinement_prompt


def load_yaml(file_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_file_content(file_path: str, required: bool = False) -> str:
    """
    Read content from a file and return it as a string.

    With required=True a missing file raises instead of returning an error
    string. Prompt inputs always pass required=True, so a prompt is never built
    with a placeholder in place of real content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(
                f"Prompt input not found: {file_path}. Refusing to build a prompt "
                f"with a placeholder in place of real content."
            )
        return f"[ERROR: File not found: {file_path}]"
    except Exception as e:
        if required:
            raise
        return f"[ERROR: Could not read file {file_path}: {str(e)}]"


EXAMPLE_VARIANTS = ('perdomain', 'stub')
# 'fixed:<path>' is also accepted: one exemplar, the same file for every domain.
FIXED_EXAMPLE_PREFIX = 'fixed:'
HINTS_VARIANTS = ('full', 'none')


def resolve_example(domain_config: dict, defaults: dict, variant: str,
                    domain_key: str = None, domains_config: dict = None) -> tuple:
    """
    Resolve the section 7 worked example for one arm.

    Returns (content, components_override, description, example_domain). A
    components_override of {'example_heuristic': False} makes the template emit its
    blind-stub section instead of the content. example_domain is the display name of
    the domain the example was written for (None for the stub and the placeholder),
    which the prompt names in section 7.

    Variants:
        perdomain    the domain's own worked example (config example_heuristic)
        stub         template's blind h=0 stub; no guidance to imitate
        fixed:<path> one named exemplar, identical for every domain (ablation arm 3)
    """
    if variant.startswith(FIXED_EXAMPLE_PREFIX):
        fixed_path = variant[len(FIXED_EXAMPLE_PREFIX):]
        if not fixed_path:
            raise ValueError(f"'{FIXED_EXAMPLE_PREFIX}' needs a path, e.g. "
                             f"'{FIXED_EXAMPLE_PREFIX}Pytrich/Heuristics/GeneratedExamples/"
                             f"snake_heuristic.py'")
        return (read_file_content(fixed_path, required=True), None,
                f'FIXED exemplar: {os.path.basename(fixed_path)}',
                example_domain_of(fixed_path, domains_config))

    if variant not in EXAMPLE_VARIANTS:
        raise ValueError(f"Unknown --example '{variant}'. Valid: "
                         f"{', '.join(EXAMPLE_VARIANTS)}, or "
                         f"'{FIXED_EXAMPLE_PREFIX}<path>'")

    path = domain_config.get('example_heuristic') or defaults['fallback_example']

    if variant == 'stub':
        return '', {'example_heuristic': False}, 'blind h=0 stub', None

    return (read_file_content(path, required=True), None, os.path.basename(path),
            example_domain_of(path, domains_config))


# Examples written for domains that have no entry in domains.yaml.
NON_BENCHMARK_EXAMPLE_DOMAINS = {'snake': 'Snake'}


def example_domain_of(path: str, domains_config: dict = None):
    """
    Display name of the domain an example heuristic was written for, from its file name,
    or None when the file name does not say.

    Examples are named <prefix>_heuristic[_example].py. The prefix is matched against
    each domain's file_prefix, then its key (depots_heuristic_example.py has prefix
    'depots' but file_prefix 'depot'), then NON_BENCHMARK_EXAMPLE_DOMAINS. Anything
    else returns None, so section 7 makes no claim about it rather than a wrong one.
    """
    name = os.path.basename(path)
    if '_heuristic' not in name:
        return None
    prefix = name.split('_heuristic')[0]
    domains = (domains_config or {}).get('domains') or {}
    for key, cfg in domains.items():
        if cfg.get('file_prefix') == prefix or key == prefix:
            return cfg['name']
    return NON_BENCHMARK_EXAMPLE_DOMAINS.get(prefix)


def resolve_hints(domain_key: str, domain_config: dict, domains_config: dict, variant: str) -> tuple:
    """
    Resolve the section 2.5 hint block for one arm.

    Returns (text, components_override, description). Variants: full, none.
    """
    if variant not in HINTS_VARIANTS:
        raise ValueError(f"Unknown --hints '{variant}'. Valid: {', '.join(HINTS_VARIANTS)}")

    if variant == 'none':
        return '', {'domain_hints': False}, 'omitted'

    return domain_config.get('hints', ''), None, 'full'


BASE_MAX_TOKENS = 16384

THINKING_LEVELS = ('off', 'low', 'medium', 'high')


def resolve_thinking(model_key: str, models_config: dict, level: str = 'off',
                     temperature: float = None) -> dict:
    """
    Translate --thinking / --temperature into a provider-agnostic request spec.

    The models in config/models.yaml expose three mutually incompatible reasoning
    APIs, so the level name is resolved through models.yaml:thinking_styles rather
    than sent verbatim. Sending the wrong one is an HTTP 400, not a warning.

    Returns a dict consumed by the call_*_api functions:
        style        one of effort | budget | reasoning_effort | gemini_budget | none
        level        the requested level, verbatim ('off' disables thinking)
        value        the provider's value: a level name, or a token budget
        max_tokens   output budget, raised by the thinking budget where the
                     provider counts thinking against max_tokens
        temperature  None means "omit the field" (the provider default)

    Raises:
        ValueError: for an unknown level, or a temperature on a model that
                    rejects sampling parameters (Opus 4.7 and later).
    """
    if level not in THINKING_LEVELS:
        raise ValueError(f"Unknown --thinking level '{level}'. Valid: {', '.join(THINKING_LEVELS)}")

    style = models_config.get('thinking_styles', {}).get(model_key, 'none')
    budgets = models_config.get('thinking_budgets', {})

    spec = {
        'style': style,
        'level': level,
        'value': None,
        'max_tokens': BASE_MAX_TOKENS,
        'temperature': temperature,
    }

    # Opus 4.7 and later reject temperature/top_p/top_k outright.
    if temperature is not None and style == 'effort':
        raise ValueError(
            f"Model '{model_key}' does not accept a temperature (adaptive-thinking models "
            f"reject sampling parameters with HTTP 400). Omit --temperature to use the default."
        )

    if level == 'off' or style == 'none':
        return spec

    if style in ('effort', 'reasoning_effort'):
        spec['value'] = level
    elif style in ('budget', 'gemini_budget'):
        budget = budgets.get(style, {}).get(level)
        if budget is None:
            raise ValueError(f"No {style} budget configured for level '{level}'")
        spec['value'] = budget
        if style == 'budget':
            # budget_tokens is counted against max_tokens and must stay strictly
            # below it, so give the generated code its full budget on top.
            spec['max_tokens'] = BASE_MAX_TOKENS + budget

    return spec


def describe_thinking(spec: dict) -> str:
    """One-line human-readable summary of a resolved thinking spec."""
    if spec['level'] == 'off':
        return 'thinking=off'
    if spec['style'] == 'none':
        return f"thinking={spec['level']} (ignored: model has no thinking control)"
    return f"thinking={spec['level']} via {spec['style']}={spec['value']}"


def get_naming(domain_name: str, model_name: str, counter: int, file_prefix: str = None,
               ablation_suffix: str = '') -> dict:
    """
    Generate all naming conventions for a heuristic.

    Args:
        domain_name: Domain name (e.g., "Towers", "Barman_BDI")
        model_name: Model name (e.g., "Gemini25Pro", "Gemini20Flash")
        counter: Heuristic counter
        file_prefix: Explicit filename prefix from domains.yaml (e.g., "blocksworld", "depot")
        ablation_suffix: Optional suffix appended to the output folder only
                         (e.g., "-A1"). Class/param/filename are unchanged so the
                         heuristic itself is name-stable across ablations.

    Returns:
        Dictionary with class_name, param_name, filename, output_folder
    """
    # Normalize domain name for different contexts
    domain_title = domain_name.replace("-", "").replace("_", "")
    model_lower = model_name.lower()

    # Use explicit file_prefix if provided, else fall back to domain_lower
    prefix = file_prefix if file_prefix else domain_name.lower().replace("-", "_")

    return {
        'class_name': f'{domain_title}{model_name}Heuristic{counter}',
        # e.g., "TowersGemini25ProHeuristic5"

        'param_name': f'{prefix}_{model_lower}_heuristic_{counter}',
        # e.g., "towers_gemini25pro_heuristic_5"

        'filename': f'{prefix}_heuristic_{counter}.py',
        # e.g., "towers_heuristic_5.py"

        'output_folder': f'Pytrich/Heuristics/GeneratedHeuristics/{domain_name}/{model_name}{ablation_suffix}'
        # e.g., "Pytrich/Heuristics/GeneratedHeuristics/Towers/Gemini25Pro" or
        #       "Pytrich/Heuristics/GeneratedHeuristics/Towers/Gemini25Pro-A1"
    }


def get_naming_versioned(domain_name: str, model_name: str, counter: int, version: int, file_prefix: str = None,
                         ablation_suffix: str = '') -> dict:
    """
    Generate naming conventions for an iterative heuristic version.

    Args:
        domain_name: Domain name (e.g., "Towers")
        model_name: Model name (e.g., "Gemini25Pro")
        counter: Heuristic index
        version: Iteration version (0 = initial, 1 = first refinement, etc.)
        file_prefix: Explicit filename prefix from domains.yaml (e.g., "blocksworld", "depot")

    Returns:
        Dictionary with class_name, param_name, filename, output_folder
    """
    domain_title = domain_name.replace("-", "").replace("_", "")
    model_lower = model_name.lower()

    # Use explicit file_prefix if provided, else fall back to domain_lower
    prefix = file_prefix if file_prefix else domain_name.lower().replace("-", "_")

    version_suffix = f"_v{version}" if version > 0 else ""

    return {
        'class_name': f'{domain_title}{model_name}Heuristic{counter}{version_suffix.replace("_", "").upper()}',
        'param_name': f'{prefix}_{model_lower}_heuristic_{counter}{version_suffix}',
        'filename': f'{prefix}_heuristic_{counter}{version_suffix}.py',
        'output_folder': f'Pytrich/Heuristics/GeneratedHeuristics/{domain_name}/{model_name}{ablation_suffix}'
    }


ABLATION_COMPONENTS = {
    'A0': {},  # baseline — all sections on (defaults)
    'A1': {'domain_hints': False},          # remove §2.5 domain hints
    'A2': {'critical_fact_format': False},  # remove §§3-5 critical fact / goal / bitwise
    'A3': {'example_heuristic': False},     # remove §7 example heuristic
    'A4': {'pytrich_api': False},           # remove §§8-9.1 Pytrich classes
    'A5': {'admissibility': False},         # remove §10 admissibility section
    'A6': {'win_lose_patterns': False},     # remove §11 win/lose patterns
    'A7': {'design_procedure': False},      # remove §12 required design procedure
    # A8 (refinement-loop ablation) is pipeline-level, handled by --count not --ablation
}


def build_base_prompt(domain_config: dict, model_config: dict, defaults: dict, naming: dict, counter: int,
                      components: dict = None, domain_key: str = None, domains_config: dict = None,
                      example_variant: str = 'perdomain', hints_variant: str = 'full') -> str:
    """
    Build the base LLM prompt for a domain (shared between one-shot and iterative).

    Args:
        components: Optional dict gating ablatable prompt sections. None = use defaults
                    (all sections on). See templates.prompt_template.DEFAULT_COMPONENTS.
        domain_key: Domain key in domains.yaml.
        domains_config: Full domains.yaml.
        example_variant: Section 7 arm — see resolve_example / EXAMPLE_VARIANTS.
        hints_variant: Section 2.5 arm — see resolve_hints / HINTS_VARIANTS.

    Returns:
        The complete base prompt string.
    """
    domain_name = domain_config['name']
    benchmark_base = defaults['benchmark_base']
    domain_folder = domain_config['folder']

    domain_file_path = f"{benchmark_base}/{domain_folder}/{domain_config['domain_file']}"
    smallest_instance_path = f"{benchmark_base}/{domain_folder}/{domain_config['smallest_instance']}"
    largest_instance_path = f"{benchmark_base}/{domain_folder}/{domain_config['largest_instance']}"

    # The domain and its instances are the irreducible core of the prompt: a
    # placeholder in place of any of them would invalidate the run outright.
    domain_content = read_file_content(domain_file_path, required=True)
    smallest_instance_content = read_file_content(smallest_instance_path, required=True)
    largest_instance_content = read_file_content(largest_instance_path, required=True)

    example_heuristic_content, example_override, _, example_domain = resolve_example(
        domain_config, defaults, example_variant, domain_key, domains_config or {})
    domain_hints, hints_override, _ = resolve_hints(
        domain_key, domain_config, domains_config or {}, hints_variant)

    # Variant overrides compose with any --ablation component gating.
    if example_override or hints_override:
        components = dict(components) if components else dict(DEFAULT_COMPONENTS)
        components.update(example_override or {})
        components.update(hints_override or {})

    return generate_prompt(
        domain_name=domain_name,
        domain_content=domain_content,
        smallest_instance_name=domain_config['smallest_instance'],
        smallest_instance_content=smallest_instance_content,
        largest_instance_name=domain_config['largest_instance'],
        largest_instance_content=largest_instance_content,
        example_heuristic_content=example_heuristic_content,
        class_name=naming['class_name'],
        param_name=naming['param_name'],
        counter=counter,
        domain_hints=domain_hints,
        components=components,
        example_domain=example_domain,
    )


def resolve_ablation_components(ablation_code: str = None) -> dict:
    """Resolve an ablation code (e.g., 'A1') to a components dict."""
    if not ablation_code or ablation_code == 'A0':
        return None  # use defaults
    if ablation_code not in ABLATION_COMPONENTS:
        raise ValueError(
            f"Unknown ablation code: {ablation_code!r}. "
            f"Valid: {sorted(ABLATION_COMPONENTS.keys())}"
        )
    from templates.prompt_template import DEFAULT_COMPONENTS
    components = dict(DEFAULT_COMPONENTS)
    components.update(ABLATION_COMPONENTS[ablation_code])
    return components


def _parse_planner_output(stdout: str) -> dict:
    """
    Parse planner stdout to extract metrics.

    Looks for the Descriptions format lines:
        Search Status : GOAL
        Nodes Expanded : 123
        Search Time (seconds) : 0.1234
        Solution Size : 42

    Returns:
        Dict with keys: status, nodes_expanded, search_time, solution_size
    """
    result = {
        'status': 'UNKNOWN',
        'nodes_expanded': 'N/A',
        'search_time': 'N/A',
        'solution_size': 'N/A',
    }

    for line in stdout.splitlines():
        line = line.strip()
        if 'Search Status' in line and ':' in line:
            result['status'] = line.split(':', 1)[1].strip()
        elif 'Nodes Expanded' in line and ':' in line:
            result['nodes_expanded'] = line.split(':', 1)[1].strip()
        elif 'Search Time (seconds)' in line and ':' in line:
            result['search_time'] = line.split(':', 1)[1].strip()
        elif 'Solution Size' in line and ':' in line:
            result['solution_size'] = line.split(':', 1)[1].strip()

    return result


def test_heuristic(heuristic_file: str, domain_config: dict, defaults: dict, timeout: int = 60) -> dict:
    """
    Test a heuristic by running the planner on the smallest instance.

    Args:
        heuristic_file: Path to the generated heuristic .py file
        domain_config: Domain configuration from domains.yaml
        defaults: Default settings from domains.yaml
        timeout: Subprocess timeout in seconds

    Returns:
        Dict with keys: status, nodes_expanded, search_time, solution_size, error
    """
    benchmark_base = defaults['benchmark_base']
    domain_folder = domain_config['folder']
    domain_file = f"{benchmark_base}/{domain_folder}/{domain_config['domain_file']}"
    instance_file = f"{benchmark_base}/{domain_folder}/{domain_config['smallest_instance']}"

    cmd = [
        sys.executable, '__main__.py',
        domain_file, instance_file,
        '--heuristic-file', heuristic_file,
        '-S', 'Astar()',
        '-ms',
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        result = _parse_planner_output(proc.stdout)
        if proc.returncode != 0 and result['status'] == 'UNKNOWN':
            # Capture stderr for error feedback
            error_text = proc.stderr.strip() if proc.stderr else ''
            if not error_text and proc.stdout:
                # Sometimes errors are in stdout
                error_text = proc.stdout.strip()[-2000:]
            result['error'] = error_text
        else:
            result['error'] = ''
        return result

    except subprocess.TimeoutExpired:
        return {
            'status': 'TIMEOUT',
            'nodes_expanded': 'N/A',
            'search_time': f'>{timeout}',
            'solution_size': 'N/A',
            'error': '',
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'nodes_expanded': 'N/A',
            'search_time': 'N/A',
            'solution_size': 'N/A',
            'error': str(e),
        }


# Cache for TDG baseline results per domain
_tdg_baseline_cache = {}


def run_tdg_baseline(domain_config: dict, defaults: dict, timeout: int = 60) -> dict:
    """
    Run TDG heuristic baseline on the smallest instance (cached per domain).

    Returns:
        Dict with same structure as test_heuristic output.
    """
    domain_key = domain_config['name']
    if domain_key in _tdg_baseline_cache:
        return _tdg_baseline_cache[domain_key]

    benchmark_base = defaults['benchmark_base']
    domain_folder = domain_config['folder']
    domain_file = f"{benchmark_base}/{domain_folder}/{domain_config['domain_file']}"
    instance_file = f"{benchmark_base}/{domain_folder}/{domain_config['smallest_instance']}"

    cmd = [
        sys.executable, '__main__.py',
        domain_file, instance_file,
        '-H', 'TDG()',
        '-S', 'Astar()',
        '-ms',
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        result = _parse_planner_output(proc.stdout)
        result['error'] = ''
    except subprocess.TimeoutExpired:
        result = {
            'status': 'TIMEOUT',
            'nodes_expanded': 'N/A',
            'search_time': f'>{timeout}',
            'solution_size': 'N/A',
            'error': '',
        }
    except Exception as e:
        result = {
            'status': 'ERROR',
            'nodes_expanded': 'N/A',
            'search_time': 'N/A',
            'solution_size': 'N/A',
            'error': str(e),
        }

    _tdg_baseline_cache[domain_key] = result
    return result


def get_api_key(provider: str, models_config: dict, args_api_key: str = None) -> str:
    """
    Get API key from args or environment variable.

    Args:
        provider: The provider name (e.g., 'google')
        models_config: The loaded models.yaml configuration
        args_api_key: Optional API key passed via command line

    Returns:
        The API key string

    Raises:
        ValueError: If no API key is found
    """
    if args_api_key:
        return args_api_key

    # Read env_vars from config instead of hardcoding
    env_vars = models_config.get('env_vars', {})

    env_var = env_vars.get(provider)
    if env_var:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key

    raise ValueError(
        f"No API key provided. Use --api-key or set {env_vars.get(provider, 'API_KEY')} environment variable."
    )


def call_gemini_api(prompt: str, api_url: str, api_key: str, filename: str, max_retries: int = 3,
                    thinking: dict = None) -> tuple:
    """
    Call the Gemini API to generate code with retry logic.

    Args:
        prompt: The prompt to send
        api_url: Full API URL with model
        api_key: API key
        filename: Expected filename for the response
        max_retries: Maximum number of retry attempts for transient failures (default: 3)

    Returns:
        Tuple of (generated_code_string, usage_dict) where usage_dict has
        'input_tokens' and 'output_tokens' keys.

    Raises:
        requests.exceptions.HTTPError: After all retries exhausted for HTTP errors
        ValueError: For unexpected response structure
        json.JSONDecodeError: For malformed JSON responses
    """
    # Send the key as a header rather than `?key=`, so it cannot leak: requests
    # echoes the full URL in HTTPError messages, which put the key on stdout and
    # into any captured log on the first 429.
    full_url = api_url

    chat_history = [
        {
            "role": "user",
            "parts": [
                {
                    "text": f"{prompt}\n\n"
                            f"Provide the response as a JSON object with two keys: "
                            f"'filename' as '{filename}' and "
                            f"'code' (string, the complete Python code). "
                            f"Ensure the filename ends with '.py'."
                }
            ]
        }
    ]

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "filename": {"type": "STRING"},
                    "code": {"type": "STRING"}
                },
                "required": ["filename", "code"]
            }
        }
    }

    if thinking:
        if thinking['style'] == 'gemini_budget' and thinking['value'] is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking['value']
            }
        if thinking.get('temperature') is not None:
            payload["generationConfig"]["temperature"] = thinking['temperature']

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # Retry logic with exponential backoff for transient errors
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(full_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()

            candidate = (result.get("candidates") or [{}])[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            # With thinkingConfig set, reasoning parts precede the answer and carry
            # `thought: true`. Taking parts[0] would hand the thinking text to
            # json.loads, or miss `text` entirely, so pick the first answer part.
            json_string = next(
                (p["text"] for p in parts if p.get("text") and not p.get("thought")),
                None,
            )

            if json_string is not None:
                code_data = json.loads(json_string)
                usage_meta = result.get("usageMetadata", {})
                usage = {
                    "input_tokens": usage_meta.get("promptTokenCount", 0),
                    "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                    "thinking_tokens": usage_meta.get("thoughtsTokenCount", 0),
                }
                return code_data.get("code", ""), usage

            raise ValueError(
                f"No answer part in Gemini response "
                f"(finishReason={candidate.get('finishReason')}, "
                f"parts={[sorted(p) for p in parts]}, "
                f"usage={result.get('usageMetadata')}). "
                f"A thinking-only response usually means the output budget was "
                f"consumed by reasoning."
            )

        except requests.exceptions.HTTPError as e:
            last_exception = e
            # Retry on 429 (rate limit), 500, 502, 503, 504 (server errors)
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff
                    print(f"    HTTP {status_code}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            raise

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff
                print(f"    Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise

    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    raise ValueError("API call failed after all retries")


def call_openai_api(prompt: str, api_url: str, api_key: str, model_id: str, filename: str, max_retries: int = 3,
                    thinking: dict = None) -> tuple:
    """
    Call the OpenAI API to generate code with retry logic.

    Args:
        prompt: The prompt to send
        api_url: API endpoint URL (e.g., https://api.openai.com/v1/chat/completions)
        api_key: API key
        model_id: Model identifier (e.g., gpt-4o)
        filename: Expected filename for the response
        max_retries: Maximum number of retry attempts for transient failures

    Returns:
        Tuple of (generated_code_string, usage_dict) where usage_dict has
        'input_tokens' and 'output_tokens' keys.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Provide the response as a JSON object with two keys: "
                    f"'filename' as '{filename}' and "
                    f"'code' (string, the complete Python code). "
                    f"Ensure the filename ends with '.py'."
                )
            }
        ],
        "response_format": {"type": "json_object"}
    }

    if thinking:
        if thinking['style'] == 'reasoning_effort' and thinking['value'] is not None:
            payload["reasoning_effort"] = thinking['value']
        if thinking.get('temperature') is not None:
            payload["temperature"] = thinking['temperature']

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            code_data = json.loads(content)
            usage_data = result.get("usage", {})
            usage = {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            }
            return code_data.get("code", ""), usage

        except requests.exceptions.HTTPError as e:
            last_exception = e
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    print(f"    HTTP {status_code}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            raise

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                print(f"    Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise

    if last_exception:
        raise last_exception
    raise ValueError("API call failed after all retries")


def _extract_json_from_text(text: str) -> dict:
    """
    Extract a JSON object from text that may contain markdown fences or extra content.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code fences (```json ... ```)
    3. Find a { ... } block containing "code" key
    4. If text contains a class definition, treat as raw code

    Args:
        text: Raw text that may contain JSON

    Returns:
        Dictionary with at least a 'code' key

    Raises:
        ValueError: If no valid JSON or code could be extracted
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find a JSON object containing "code" key
    brace_match = re.search(r'\{[^{}]*"code"[^{}]*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3b: Try finding outermost braces (for nested JSON)
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    candidate = json.loads(text[start:i + 1])
                    if "code" in candidate:
                        return candidate
                except json.JSONDecodeError:
                    pass
                start = None

    # Strategy 4: If response contains a class definition, treat entire text as code
    if 'class ' in text and 'def __call__' in text:
        return {"code": text}

    raise ValueError(f"Could not extract JSON or code from response (length={len(text)})")


def call_anthropic_api(prompt: str, api_url: str, api_key: str, model_id: str, filename: str, max_retries: int = 3,
                       thinking: dict = None) -> tuple:
    """
    Call the Anthropic API to generate code with retry logic.

    Args:
        prompt: The prompt to send
        api_url: API endpoint URL (e.g., https://api.anthropic.com/v1/messages)
        api_key: API key
        model_id: Model identifier (e.g., claude-sonnet-4-5-20250929)
        filename: Expected filename for the response
        max_retries: Maximum number of retry attempts for transient failures

    Returns:
        Tuple of (generated_code_string, usage_dict) where usage_dict has
        'input_tokens' and 'output_tokens' keys.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model_id,
        "max_tokens": (thinking or {}).get('max_tokens', BASE_MAX_TOKENS),
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Provide the response as a JSON object with two keys: "
                    f"'filename' as '{filename}' and "
                    f"'code' (string, the complete Python code). "
                    f"Ensure the filename ends with '.py'."
                )
            }
        ]
    }

    if thinking and thinking['level'] != 'off':
        if thinking['style'] == 'effort':
            # Opus 4.6+: adaptive thinking, depth controlled by effort. budget_tokens
            # and temperature are rejected on 4.7 and later, so neither is sent.
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": thinking['value']}
        elif thinking['style'] == 'budget':
            # Opus 4 / Sonnet 4.5: fixed budget, must stay under max_tokens.
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking['value']}

    if thinking and thinking.get('temperature') is not None:
        payload["temperature"] = thinking['temperature']

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            # With thinking enabled the response leads with one or more thinking
            # blocks, so the text block is not necessarily content[0].
            text = next(
                (b["text"] for b in result.get("content", []) if b.get("type") == "text"),
                None,
            )
            if text is None:
                raise ValueError(
                    f"No text block in Anthropic response (stop_reason="
                    f"{result.get('stop_reason')!r}, block types="
                    f"{[b.get('type') for b in result.get('content', [])]})"
                )
            code_data = _extract_json_from_text(text)
            usage_data = result.get("usage", {})
            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
            }
            return code_data.get("code", ""), usage

        except requests.exceptions.HTTPError as e:
            last_exception = e
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                # 529 is Anthropic's overload status code
                if status_code in (429, 500, 502, 503, 504, 529) and attempt < max_retries:
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    print(f"    HTTP {status_code}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            raise

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                print(f"    Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise

    if last_exception:
        raise last_exception
    raise ValueError("API call failed after all retries")


def call_llm_api(prompt: str, model_config: dict, api_key: str, filename: str, max_retries: int = 3,
                 thinking: dict = None) -> tuple:
    """
    Dispatch an API call to the appropriate provider.

    Args:
        prompt: The prompt to send
        model_config: Model configuration dict (must include 'provider', 'api_base', 'model_id')
        api_key: API key for the provider
        filename: Expected filename for the response
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (generated_code_string, usage_dict) where usage_dict has
        'input_tokens' and 'output_tokens' keys.

    Raises:
        ValueError: For unknown providers
    """
    provider = model_config['provider']
    api_base = model_config['api_base']
    model_id = model_config['model_id']

    if provider == "google":
        api_url = f"{api_base}/{model_id}:generateContent"
        code, usage = call_gemini_api(prompt, api_url, api_key, filename, max_retries, thinking)
        return postprocess_code(code, usage)

    elif provider == "openai":
        code, usage = call_openai_api(prompt, api_base, api_key, model_id, filename, max_retries, thinking)
        return postprocess_code(code, usage)

    elif provider == "anthropic":
        code, usage = call_anthropic_api(prompt, api_base, api_key, model_id, filename, max_retries, thinking)
        return postprocess_code(code, usage)

    else:
        raise ValueError(f"Unknown provider '{provider}'. Supported: google, openai, anthropic")


def postprocess_code(code: str, usage: dict) -> tuple:
    """Repair transport-level escaping and syntax-check, recording both in `usage`.

    Models sometimes return the code field double-escaped, so `json.loads` yields a
    string holding a literal backslash-n rather than a newline. The result is a
    one-line file that cannot parse. Observed on gemini-3-flash for Depots while
    Towers and Rover from the same model were fine, so it is intermittent.

    This is a serialization artifact, not a code-quality defect, and repairing it
    is therefore correct -- but the paper measures how often models emit *valid*
    code, so a silent repair would inflate that rate. Both flags travel in `usage`
    into api_costs.csv so the two failure kinds stay separable:

        escaping_repaired=1  our transport fix, not a model code error
        syntax_ok=0          the model really did emit unparseable Python

    A syntax failure is recorded, warned about, and still written to disk: the file
    is the experimental datum.
    """
    repaired = False
    if code and '\\n' in code and '\n' not in code:
        # The string is a JSON string body that went through one escaping round too
        # few, so decode it as exactly that rather than hand-replacing escapes,
        # which would mangle legitimate backslashes in the code. raw_decode rather
        # than loads because the observed case also carried the response object's
        # own closing `"}` inside the value; raw_decode stops at the string's real
        # end and reports where, so the tail can be dropped instead of poisoning
        # the file.
        wrapped = f'"{code}"'
        try:
            code, end = json.JSONDecoder().raw_decode(wrapped)
            tail = wrapped[end:]
        except ValueError:
            code = code.replace('\\n', '\n').replace('\\t', '\t')
            tail = ''
        repaired = True
        print("  NOTE: response was double-escaped (literal \\n, no real newlines); "
              "un-escaped before saving. Logged as escaping_repaired.")
        if tail:
            print(f"  NOTE: dropped {len(tail)} trailing char(s) after the code string.")

    # The dominant defect in the existing corpus: the response object's closing
    # quote and brace arrive as trailing lines after otherwise complete code (27 of
    # 31 unparseable files in GeneratedHeuristics/ are only this). Strip it only
    # when the code does not already parse AND stripping makes it parse, so code
    # that is fine is never touched and a real syntax error is never masked.
    stripped_debris = False
    if code and not _parses(code):
        candidate = _without_envelope_debris(code)
        if candidate != code and _parses(candidate):
            code, stripped_debris = candidate, True
            print("  NOTE: dropped trailing JSON-envelope line(s) after the code; "
                  "the code itself was complete. Logged as envelope_stripped.")

    syntax_ok = True
    try:
        ast.parse(code)
    except SyntaxError as e:
        syntax_ok = False
        print(f"  WARNING: generated code does not parse (line {e.lineno}: {e.msg}). "
              f"Saving anyway -- the file is the datum -- but it will not run.")

    usage['escaping_repaired'] = repaired
    usage['envelope_stripped'] = stripped_debris
    usage['syntax_ok'] = syntax_ok
    return code, usage


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _without_envelope_debris(code: str) -> str:
    """Drop trailing lines that are only JSON/markdown envelope characters.

    A line whose entire content is `"`, `}`, `"}` or a code fence is never valid
    Python at module level, so removing it cannot discard real code.
    """
    lines = code.splitlines()
    end = len(lines)
    while end > 0 and lines[end - 1].strip() in ('"', '}', '"}', '```', ''):
        end -= 1
    return '\n'.join(lines[:end]) + '\n' if end < len(lines) else code


def compute_cost(usage: dict, model_config: dict) -> float:
    """Compute dollar cost from token usage and model pricing.

    Reasoning tokens are billed at the output rate, but providers report them
    differently: Gemini's `candidatesTokenCount` EXCLUDES `thoughtsTokenCount`,
    while Anthropic and OpenAI already fold reasoning into their output count.
    So only the Gemini branch sets 'thinking_tokens', and adding it here is what
    keeps the two comparable -- don't "fix" the other providers by populating it,
    that would double-count.
    """
    input_cost = model_config.get('input_cost_per_mtok', 0)
    output_cost = model_config.get('output_cost_per_mtok', 0)
    # Gemini Pro tiers on prompt length: past the threshold BOTH rates step up.
    threshold = model_config.get('long_prompt_threshold_tokens')
    if threshold and usage['input_tokens'] > threshold:
        input_cost = model_config.get('long_input_cost_per_mtok', input_cost)
        output_cost = model_config.get('long_output_cost_per_mtok', output_cost)
    billed_output = usage['output_tokens'] + usage.get('thinking_tokens', 0)
    return (usage['input_tokens'] * input_cost + billed_output * output_cost) / 1_000_000


def _blob_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()


def _prompt_context_files(domain_config: dict, defaults: dict) -> list:
    """The three benchmark files build_base_prompt embeds verbatim, as (path, text)."""
    base = defaults['benchmark_base']
    folder = domain_config['folder']
    out = []
    for key in ('domain_file', 'smallest_instance', 'largest_instance'):
        name = domain_config.get(key)
        if not name:
            continue
        path = f"{base}/{folder}/{name}"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                out.append((path, f.read()))
        except OSError:
            continue
    return out


def build_prompt_log(prompt: str, domain_config: dict, defaults: dict, meta: dict) -> str:
    """A commented record of the prompt, to append to the generated heuristic.

    Written so a heuristic file answers "what produced this?" on its own. Working
    that out retrospectively for the 2026 batches took a day of git forensics --
    reflog timestamps and stash-only commits -- because nothing was recorded at
    generation time.

    The three benchmark files are replaced by path + SHA-256 + byte count rather
    than inlined: they are 95% of Rover's 410 KB prompt, they already live in the
    repo, and every candidate in a batch embeds the identical copy. The prompt's
    own SHA-256 is of the FULL text before elision, so the exact bytes remain
    verifiable. Everything that actually varies between arms -- instructions,
    hints, worked example -- is kept verbatim.
    """
    full_sha, full_len = _blob_sha256(prompt), len(prompt)
    lines = prompt.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    elided = []
    for path, text in _prompt_context_files(domain_config, defaults):
        # The embedded copy is NOT byte-identical to the file: the template runs
        # textwrap.dedent, which normalizes whitespace-only lines, so a file with a
        # line containing a single space (Towers/domain.hddl) never matches as a
        # substring. Match line-wise on rstripped text and splice by position.
        want = [ln.rstrip() for ln in text.strip().split('\n')]
        if not want or not text:
            continue
        hay = [ln.rstrip() for ln in lines]
        span = next((i for i in range(len(hay) - len(want) + 1)
                     if hay[i:i + len(want)] == want), None)
        if span is None:
            continue
        lines[span:span + len(want)] = [
            f"[FILE: {path}",
            f" sha256 {_blob_sha256(text)}, {len(text):,} bytes -- elided, see repo]",
        ]
        elided.append(path)
    body = '\n'.join(lines)

    head = [
        "=" * 74,
        "PROMPT THAT GENERATED THIS HEURISTIC",
    ]
    for k in ('model', 'version', 'thinking', 'temperature', 'arm', 'ablation',
              'run_tag', 'template_sha256', 'generated'):
        if meta.get(k) is not None:
            head.append(f"  {k}: {meta[k]}")
    head += [
        f"  prompt sha256: {full_sha}",
        f"  prompt bytes: {full_len:,} ({len(elided)} benchmark file(s) elided below)",
        "-" * 74,
    ]
    lines = head + body.replace('\r\n', '\n').replace('\r', '\n').split('\n') + ["=" * 74]
    # Comment every line: a comment can hold any text except a newline, so this
    # cannot break the file however odd the prompt is.
    return "\n\n" + "\n".join(f"# {ln}".rstrip() for ln in lines) + "\n"


def prompt_log_meta(model_config: dict, thinking: dict, example_variant: str,
                    hints_variant: str, ablation_suffix: str, run_tag: str = None) -> dict:
    """Metadata half of the prompt log: what was requested, and from which code."""
    tmpl = PROJECT_ROOT / 'templates' / 'prompt_template.py'
    try:
        template_sha = _blob_sha256(tmpl.read_text(encoding='utf-8'))
    except OSError:
        template_sha = 'unavailable'
    t = thinking or {}
    level, style = t.get('level', 'off'), t.get('style', 'none')
    budget = t.get('budget')
    return {
        'model': model_config.get('model_id', '?'),
        'thinking': f"{level} (style={style}{f', budget={budget}' if budget else ''})",
        'temperature': t.get('temperature') if t.get('temperature') is not None
                       else 'provider default (not recorded)',
        'arm': f"--example {example_variant} --hints {hints_variant}",
        'ablation': ablation_suffix or 'A0',
        'run_tag': run_tag,
        'template_sha256': template_sha,
        'generated': datetime.now().isoformat(timespec='seconds'),
    }


def format_usage(usage: dict, cost: float) -> str:
    """One-line token/cost report. Shows reasoning separately when there is any."""
    thinking = usage.get('thinking_tokens', 0)
    thinking_part = f" / {thinking:,} reasoning" if thinking else ""
    return (f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
            f"{thinking_part} | Cost: ${cost:.4f}")


def log_cost_to_csv(domain: str, model: str, heuristic_index: int, version: str,
                    usage: dict, cost: float):
    """Append a cost entry to the API costs CSV file."""
    csv_path = PROJECT_ROOT / 'Experiments' / 'outputs' / 'csv-files' / 'api_costs.csv'
    os.makedirs(csv_path.parent, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'domain', 'model', 'heuristic_index',
                             'version', 'input_tokens', 'output_tokens',
                             'thinking_tokens', 'cost_usd',
                             'escaping_repaired', 'envelope_stripped', 'syntax_ok'])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            domain, model, heuristic_index, version,
            usage['input_tokens'], usage['output_tokens'],
            usage.get('thinking_tokens', 0), f"{cost:.6f}",
            int(usage.get('escaping_repaired', False)),
            int(usage.get('envelope_stripped', False)),
            int(usage.get('syntax_ok', True)),
        ])


def register_in_constants(file_path: str, naming: dict, domain_name: str, model_name: str):
    """
    Safely register a generated heuristic in Pytrich/constants.py.

    1. Extract class name from generated file
    2. Validate the import works (subprocess check)
    3. Append import + dict entry to constants.py if validation passes
    4. Skip if already registered (duplicate check)
    """
    constants_path = PROJECT_ROOT / 'Pytrich' / 'constants.py'

    # Step 1: Extract class name from generated file
    class_name = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'class\s+(\w+)\s*\(\s*Heuristic\s*\)', line)
                if match:
                    class_name = match.group(1)
                    break
    except Exception as e:
        print(f"  Registration skipped: could not read {file_path}: {e}")
        return False

    if not class_name:
        print(f"  Registration skipped: no class inheriting from Heuristic found in {file_path}")
        return False

    # Build module path and alias
    filename_stem = Path(file_path).stem
    module_path = f"Pytrich.Heuristics.GeneratedHeuristics.{domain_name}.{model_name}.{filename_stem}"
    alias = naming['class_name']
    param_name = naming['param_name']

    # Step 2: Check for duplicates
    try:
        constants_content = constants_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  Registration skipped: could not read constants.py: {e}")
        return False

    if param_name in constants_content:
        print(f"  Registration skipped: '{param_name}' already in constants.py")
        return False

    # Step 3: Validate the import works
    import_stmt = f"from {module_path} import {class_name}"
    result = subprocess.run(
        [sys.executable, "-c", import_stmt],
        capture_output=True, text=True, timeout=15
    )

    if result.returncode != 0:
        print(f"  Registration skipped: import validation failed")
        print(f"    {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}")
        return False

    # Step 4: Append import + dict entry to constants.py
    # Import line goes before HEURISTICS = {
    import_line = f"from .Heuristics.GeneratedHeuristics.{domain_name}.{model_name}.{filename_stem} import {class_name} as {alias}"
    dict_entry = f'    "{param_name}": {alias},'

    lines = constants_content.splitlines(keepends=True)
    new_lines = []
    import_inserted = False
    dict_inserted = False

    for i, line in enumerate(lines):
        # Insert import just before HEURISTICS = {
        if not import_inserted and line.strip().startswith('HEURISTICS = {'):
            new_lines.append(f"# Auto-registered by generate_heuristic.py\n")
            new_lines.append(import_line + "\n")
            new_lines.append("\n")
            import_inserted = True

        # Insert dict entry just before AGGREGATIONS = {
        if not dict_inserted and line.strip().startswith('AGGREGATIONS = {'):
            # Go back to find the closing } of HEURISTICS and insert before it
            # Insert before the current line (which is AGGREGATIONS)
            # We need to insert before the } that closes HEURISTICS
            # Look backwards for the }
            j = len(new_lines) - 1
            while j >= 0 and new_lines[j].strip() not in ('}', '}\\n'):
                j -= 1
            if j >= 0:
                new_lines.insert(j, dict_entry + "\n")
                dict_inserted = True

        new_lines.append(line)

    if import_inserted and dict_inserted:
        constants_path.write_text(''.join(new_lines), encoding='utf-8')
        print(f"  Registered in constants.py: {param_name} -> {alias}")
        return True
    else:
        print(f"  Registration skipped: could not find insertion points in constants.py")
        return False


def generate_for_domain(
    domain_key: str,
    domain_config: dict,
    model_config: dict,
    defaults: dict,
    counter: int,
    api_key: str,
    dry_run: bool = False,
    max_retries: int = 3,
    ablation_components: dict = None,
    ablation_suffix: str = '',
    thinking: dict = None,
    domains_config: dict = None,
    example_variant: str = 'perdomain',
    hints_variant: str = 'full',
) -> bool:
    """
    Generate a single heuristic for a domain (one-shot, no feedback).

    Args:
        domain_key: Key identifier for the domain
        domain_config: Domain configuration from domains.yaml
        model_config: Model configuration from models.yaml
        defaults: Default settings from domains.yaml
        counter: Heuristic counter/index
        api_key: API key for the LLM provider
        dry_run: If True, print prompt without making API call
        max_retries: Maximum number of retries for failed API calls

    Returns:
        Tuple of (success: bool, usage: dict or None) where usage contains
        'input_tokens', 'output_tokens', and 'cost' keys on success.
    """
    domain_name = domain_config['name']
    model_name = model_config['name']
    file_prefix = domain_config.get('file_prefix')

    naming = get_naming(domain_name, model_name, counter, file_prefix, ablation_suffix)
    prompt = build_base_prompt(domain_config, model_config, defaults, naming, counter, ablation_components,
                               domain_key, domains_config, example_variant, hints_variant)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN: {domain_name} - {model_name} - Heuristic {counter}")
        print(f"{'='*60}")
        print(f"Class name: {naming['class_name']}")
        print(f"Param name: {naming['param_name']}")
        print(f"Filename: {naming['filename']}")
        print(f"Output folder: {naming['output_folder']}")
        print(f"\nPrompt length: {len(prompt)} characters")
        print(f"\nPrompt preview (first 2000 chars):")
        print("-" * 40)
        print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
        print("-" * 40)
        return True, None

    print(f"\nGenerating {naming['class_name']}...")

    try:
        code, usage = call_llm_api(prompt, model_config, api_key, naming['filename'], max_retries, thinking)

        output_folder = naming['output_folder']
        os.makedirs(output_folder, exist_ok=True)

        file_path = os.path.join(output_folder, naming['filename'])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
            f.write(build_prompt_log(prompt, domain_config, defaults, prompt_log_meta(
                model_config, thinking, example_variant, hints_variant,
                ablation_suffix, run_tag=ablation_suffix.lstrip('-') or None)))

        cost = compute_cost(usage, model_config)
        usage['cost'] = cost
        print(f"  Successfully saved to: {file_path}")
        print(f"  {format_usage(usage, cost)}")
        log_cost_to_csv(domain_name, model_name, counter, "batch", usage, cost)
        register_in_constants(file_path, naming, domain_name, model_name)
        return True, usage

    except requests.exceptions.HTTPError as e:
        print(f"  HTTP error: {e}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
        return False, None
    except Exception as e:
        print(f"  Error: {e}")
        return False, None


def generate_iterative(
    domain_key: str,
    domain_config: dict,
    model_config: dict,
    defaults: dict,
    counter: int,
    iterations: int,
    api_key: str,
    max_retries: int = 3,
    delay: float = 0,
    ablation_components: dict = None,
    ablation_suffix: str = '',
    thinking: dict = None,
    domains_config: dict = None,
    example_variant: str = 'perdomain',
    hints_variant: str = 'full',
) -> dict:
    """
    Generate a heuristic with iterative refinement.

    Generates v0, tests it, feeds results back to the LLM, generates v1, etc.

    Args:
        domain_key: Key identifier for the domain
        domain_config: Domain configuration from domains.yaml
        model_config: Model configuration from models.yaml
        defaults: Default settings from domains.yaml
        counter: Heuristic starting index
        iterations: Number of iterations (1 = one-shot, N = v0..vN-1)
        api_key: API key for the LLM provider
        max_retries: Maximum number of retries for failed API calls
        delay: Delay between API calls

    Returns:
        Dict with 'versions' list of {file_path, test_results, version} dicts
        and 'total_usage' with accumulated token counts and cost.
    """
    domain_name = domain_config['name']
    model_name = model_config['name']
    instance_name = domain_config['smallest_instance']
    file_prefix = domain_config.get('file_prefix')

    versions = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0,
                   'cost': 0.0, 'api_calls': 0}

    # --- v0: initial generation (same as one-shot) ---
    naming_v0 = get_naming_versioned(domain_name, model_name, counter, version=0, file_prefix=file_prefix,
                                     ablation_suffix=ablation_suffix)
    base_prompt = build_base_prompt(domain_config, model_config, defaults, naming_v0, counter, ablation_components,
                                    domain_key, domains_config, example_variant, hints_variant)

    print(f"\n  [Iteration 0/{iterations-1}] Generating {naming_v0['class_name']}...")

    try:
        code, usage = call_llm_api(base_prompt, model_config, api_key, naming_v0['filename'], max_retries, thinking)
    except Exception as e:
        print(f"    Failed to generate v0: {e}")
        return {'versions': versions, 'total_usage': total_usage}

    cost = compute_cost(usage, model_config)
    total_usage['input_tokens'] += usage['input_tokens']
    total_usage['output_tokens'] += usage['output_tokens']
    total_usage['thinking_tokens'] += usage.get('thinking_tokens', 0)
    total_usage['cost'] += cost
    total_usage['api_calls'] += 1
    print(f"    {format_usage(usage, cost)}")
    log_cost_to_csv(domain_name, model_name, counter, "v0", usage, cost)

    output_folder = naming_v0['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    file_path_v0 = os.path.join(output_folder, naming_v0['filename'])
    iter_meta = prompt_log_meta(model_config, thinking, example_variant, hints_variant,
                                ablation_suffix, run_tag=ablation_suffix.lstrip('-') or None)
    with open(file_path_v0, 'w', encoding='utf-8') as f:
        f.write(code)
        f.write(build_prompt_log(base_prompt, domain_config, defaults,
                                 dict(iter_meta, version='v0 (initial)')))
    print(f"    Saved: {file_path_v0}")
    register_in_constants(file_path_v0, naming_v0, domain_name, model_name)

    versions.append({'file_path': file_path_v0, 'version': 0, 'test_results': None})

    if iterations <= 1:
        return {'versions': versions, 'total_usage': total_usage}

    # Test v0
    print(f"    Testing v0 on {instance_name}...")
    test_results = test_heuristic(file_path_v0, domain_config, defaults)
    versions[-1]['test_results'] = test_results
    _print_test_results(test_results, "v0")

    # Get TDG baseline (cached)
    print(f"    Running TDG baseline on {instance_name}...")
    baseline = run_tdg_baseline(domain_config, defaults)
    _print_test_results(baseline, "TDG baseline")

    prev_code = code
    prev_results = test_results

    # --- v1..vN-1: iterative refinement ---
    for v in range(1, iterations):
        if delay > 0:
            time.sleep(delay)

        naming_v = get_naming_versioned(domain_name, model_name, counter, version=v, file_prefix=file_prefix,
                                        ablation_suffix=ablation_suffix)

        print(f"\n  [Iteration {v}/{iterations-1}] Generating {naming_v['class_name']}...")

        refinement_prompt = generate_refinement_prompt(
            base_prompt=base_prompt,
            iteration=v - 1,
            previous_code=prev_code,
            test_results=prev_results,
            baseline_results=baseline,
            instance_name=instance_name,
            new_class_name=naming_v['class_name'],
            new_param_name=naming_v['param_name'],
        )

        try:
            code, usage = call_llm_api(refinement_prompt, model_config, api_key, naming_v['filename'], max_retries, thinking)
        except Exception as e:
            print(f"    Failed to generate v{v}: {e}")
            break

        cost = compute_cost(usage, model_config)
        total_usage['input_tokens'] += usage['input_tokens']
        total_usage['output_tokens'] += usage['output_tokens']
        total_usage['thinking_tokens'] += usage.get('thinking_tokens', 0)
        total_usage['cost'] += cost
        total_usage['api_calls'] += 1
        print(f"    {format_usage(usage, cost)}")
        log_cost_to_csv(domain_name, model_name, counter, f"v{v}", usage, cost)

        file_path_v = os.path.join(output_folder, naming_v['filename'])
        with open(file_path_v, 'w', encoding='utf-8') as f:
            f.write(code)
            # The refinement prompt, not the base: it carries the test feedback that
            # actually shaped this version, which is the whole point of logging it.
            f.write(build_prompt_log(refinement_prompt, domain_config, defaults,
                                     dict(iter_meta, version=f'v{v} (refinement)')))
        print(f"    Saved: {file_path_v}")
        register_in_constants(file_path_v, naming_v, domain_name, model_name)

        # Test this version
        print(f"    Testing v{v} on {instance_name}...")
        test_results = test_heuristic(file_path_v, domain_config, defaults)
        _print_test_results(test_results, f"v{v}")

        versions.append({'file_path': file_path_v, 'version': v, 'test_results': test_results})

        prev_code = code
        prev_results = test_results

    return {'versions': versions, 'total_usage': total_usage}


def _print_test_results(results: dict, label: str):
    """Print test results in a compact format."""
    status = results.get('status', 'UNKNOWN')
    nodes = results.get('nodes_expanded', 'N/A')
    time_s = results.get('search_time', 'N/A')
    sol = results.get('solution_size', 'N/A')
    error = results.get('error', '')

    print(f"      [{label}] Status: {status}, Nodes: {nodes}, Time: {time_s}s, Solution: {sol}")
    if error:
        # Truncate long errors
        error_preview = error[:300] + "..." if len(error) > 300 else error
        print(f"      [{label}] Error: {error_preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate domain-dependent heuristics using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 5 independent heuristics (one-shot, no feedback)
    python generate_heuristic.py --domain towers --model gemini25pro --batch 5

    # Iterative refinement: generate v0, test, refine to v1, v2
    python generate_heuristic.py --domain towers --model claude --count 3

    # Export full prompt to file for use in any AI tool
    python generate_heuristic.py --domain towers --model gemini25pro --output-prompt /tmp/prompt.txt

    # Dry-run: preview prompt without API call
    python generate_heuristic.py --domain towers --model gemini25pro --dry-run
        """
    )

    parser.add_argument(
        '--domain', '-d',
        help='Domain name (e.g., towers, barman) or "all" for all domains'
    )
    parser.add_argument(
        '--model', '-m',
        help='Model name (e.g., gemini25pro, gemini20flash)'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1,
        help='Number of iterative refinement steps (default: 1 = one-shot). '
             'With --count N, generates v0, tests it, refines to v1, ..., vN-1.'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=None,
        help='Generate N independent heuristics (one-shot, no feedback). '
             'This is the old --count behavior.'
    )
    parser.add_argument(
        '--run-tag',
        type=str,
        default='',
        help='Suffix appended to the output folder, e.g. --run-tag arm1-no-example writes to '
             '<Domain>/<Model>-arm1-no-example. Use one tag per experimental arm so arms do '
             'not overwrite each other. Composes with --ablation.'
    )
    parser.add_argument(
        '--example',
        default='perdomain',
        metavar='VARIANT',
        help="Section 7 worked-example arm. perdomain: the domain's own example "
             "(default). stub: blind h=0 template, no guidance (ablation arms 1 and 2). "
             "fixed:<path>: one named exemplar, identical for every domain (ablation arm 3)."
    )
    parser.add_argument(
        '--hints',
        choices=list(HINTS_VARIANTS),
        default='full',
        help='Section 2.5 hint arm. full: config hints (default). none: omit '
             '(ablation arms 1 and 3).'
    )
    parser.add_argument(
        '--thinking',
        choices=list(THINKING_LEVELS),
        default='off',
        help='Reasoning/thinking depth, resolved per model via models.yaml:thinking_styles '
             '(effort level, token budget, or reasoning_effort depending on the model). '
             'Default: off.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature. Omit for the provider default (recommended). '
             'Rejected with HTTP 400 by adaptive-thinking models such as Opus 4.7.'
    )
    parser.add_argument(
        '--api-key', '-k',
        help='API key (or set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print prompt without calling API'
    )
    parser.add_argument(
        '--output-prompt',
        type=str,
        metavar='FILE',
        help='Write the complete assembled prompt to FILE and exit. '
             'Use this to feed the prompt to any other tool.'
    )
    parser.add_argument(
        '--start-index', '-s',
        type=int,
        default=0,
        help='Starting index for heuristic numbering (default: 0)'
    )
    parser.add_argument(
        '--list-domains',
        action='store_true',
        help='List all available domains and exit'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--max-retries', '-r',
        type=int,
        default=3,
        help='Maximum number of retries for failed API calls (default: 3)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0,
        help='Delay in seconds between API calls to avoid rate limits (default: 0)'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        default=None,
        choices=sorted(ABLATION_COMPONENTS.keys()),
        help='Prompt ablation code (A0=baseline, A1=no hints, A2=no critical fact format, '
             'A3=no example heuristic, A4=no Pytrich API, A5=no admissibility, '
             'A6=no win/lose patterns, A7=no design procedure). When set to non-A0, '
             'output goes to <Domain>/<Model>-<code>/ instead of <Domain>/<Model>/.'
    )

    args = parser.parse_args()
    ablation_components = resolve_ablation_components(args.ablation)
    ablation_suffix = f'-{args.ablation}' if args.ablation and args.ablation != 'A0' else ''
    if args.run_tag:
        ablation_suffix += f'-{args.run_tag}'

    # Load configurations
    config_dir = PROJECT_ROOT / 'config'
    domains_config = load_yaml(config_dir / 'domains.yaml')
    models_config = load_yaml(config_dir / 'models.yaml')

    # Handle list commands
    if args.list_domains:
        print("Available domains:")
        for key, config in domains_config['domains'].items():
            example = "has example" if config.get('example_heuristic') else "no example"
            print(f"  {key}: {config['name']} ({example})")
        return

    if args.list_models:
        print("Available models:")
        for key, config in models_config['models'].items():
            print(f"  {key}: {config['name']} ({config['model_id']})")
        return

    # Validate required arguments for generation
    if not args.domain:
        print("Error: --domain is required for generation")
        sys.exit(1)
    if not args.model:
        print("Error: --model is required for generation")
        sys.exit(1)

    # Validate model
    if args.model not in models_config['models']:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(models_config['models'].keys())}")
        sys.exit(1)

    model_config = models_config['models'][args.model]

    # Resolve --thinking/--temperature against this model's reasoning API. Fails
    # fast on an incompatible combination rather than at the first HTTP 400.
    try:
        thinking = resolve_thinking(args.model, models_config, args.thinking, args.temperature)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if thinking['level'] != 'off' or thinking['temperature'] is not None:
        temp_note = '' if thinking['temperature'] is None else f", temperature={thinking['temperature']}"
        print(f"  Generation settings: {describe_thinking(thinking)}{temp_note}")

    print(f"  Prompt arm: example={args.example}, hints={args.hints}")
    # --example lost its argparse `choices` when fixed:<path> was added, so validate
    # here: a typo should fail before the first API call, not on prompt assembly.
    if (args.example not in EXAMPLE_VARIANTS
            and not args.example.startswith(FIXED_EXAMPLE_PREFIX)):
        parser.error(f"unknown --example '{args.example}'. Valid: "
                     f"{', '.join(EXAMPLE_VARIANTS)}, or {FIXED_EXAMPLE_PREFIX}<path>")
    if args.example.startswith(FIXED_EXAMPLE_PREFIX):
        fixed = args.example[len(FIXED_EXAMPLE_PREFIX):]
        if not fixed:
            parser.error(f"--example {FIXED_EXAMPLE_PREFIX}<path> needs a path, e.g. "
                         f"{FIXED_EXAMPLE_PREFIX}Pytrich/Heuristics/GeneratedExamples/"
                         f"snake_heuristic.py")
        if not os.path.exists(fixed):
            parser.error(f"--example {FIXED_EXAMPLE_PREFIX} path does not exist: {fixed}")
        print(f"  Fixed exemplar: {fixed} (identical for every domain)")

    # Determine which domains to process
    if args.domain.lower() == 'all':
        domains_to_process = list(domains_config['domains'].keys())
    else:
        if args.domain not in domains_config['domains']:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available domains: {', '.join(domains_config['domains'].keys())}")
            sys.exit(1)
        domains_to_process = [args.domain]

    defaults = domains_config['defaults']

    # Resolve every domain's hint block now, so a bad --hints value fails before any domain is paid for.
    for domain_key in domains_to_process:
        try:
            resolve_hints(domain_key, domains_config['domains'][domain_key], domains_config, args.hints)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # --- Handle --output-prompt: write full prompt to file and exit ---
    if args.output_prompt:
        domain_key = domains_to_process[0]
        domain_config = domains_config['domains'][domain_key]
        naming = get_naming(domain_config['name'], model_config['name'], args.start_index, domain_config.get('file_prefix'),
                            ablation_suffix=ablation_suffix)
        prompt = build_base_prompt(domain_config, model_config, defaults, naming, args.start_index, ablation_components,
                                   domain_key, domains_config, args.example, args.hints)

        # Build paths for execution instructions
        benchmark_base = defaults['benchmark_base']
        domain_folder = domain_config['folder']
        save_path = os.path.join(naming['output_folder'], naming['filename'])
        domain_file = f"{benchmark_base}/{domain_folder}/{domain_config['domain_file']}"
        smallest_instance = f"{benchmark_base}/{domain_folder}/{domain_config['smallest_instance']}"

        execution_instructions = f"""

---

## Execution Instructions

Save the generated heuristic to:
`{save_path}`

After saving, test it by running:
```bash
python __main__.py {domain_file} {smallest_instance} \\
    --heuristic-file {save_path} \\
    -S "Astar()" -ms
```

**Expected output metrics:**
- `Search Status : GOAL` means it solved the problem
- `Nodes Expanded : N` — lower is better
- `Search Time (seconds) : T` — lower is better

**If the heuristic crashes or times out**, read the error output and fix the code. Common issues:
- SyntaxError → fix the Python syntax
- Timeout → the `__call__` method is too slow or heuristic values don't guide search
- Runtime error → check fact parsing, bitwise operations, and edge cases

**Iterative refinement:** After fixing any issues, re-run the test command above. Repeat until the heuristic solves the problem. Then try harder instances from the `{benchmark_base}/{domain_folder}/` directory.

**Output format:** Provide only the complete Python code for the heuristic file. No JSON wrapper needed.
"""

        output_path = os.path.abspath(args.output_prompt)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
            f.write(execution_instructions)

        print(f"Prompt written to: {output_path}")
        print(f"Prompt length: {len(prompt) + len(execution_instructions)} characters")
        print(f"\nThe prompt can now be given to any model or tool.")
        return

    # Get API key (only if not dry-run)
    api_key = None
    if not args.dry_run:
        try:
            api_key = get_api_key(model_config['provider'], models_config, args.api_key)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Determine mode: --batch (independent) or --count (iterative)
    use_batch_mode = args.batch is not None

    total_success = 0
    total_failed = 0
    cumulative_usage = {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0,
                        'cost': 0.0, 'api_calls': 0}

    for domain_key in domains_to_process:
        domain_config = domains_config['domains'][domain_key]
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain_config['name']}")
        print(f"Model: {model_config['name']}")

        if use_batch_mode:
            # --- Batch mode: N independent heuristics (old --count behavior) ---
            batch_count = args.batch
            print(f"Mode: Batch (generating {batch_count} independent heuristics)")
            print(f"{'='*60}")

            for i in range(args.start_index, args.start_index + batch_count):
                success, usage = generate_for_domain(
                    domain_key=domain_key,
                    domain_config=domain_config,
                    model_config=model_config,
                    defaults=defaults,
                    counter=i,
                    api_key=api_key,
                    dry_run=args.dry_run,
                    max_retries=args.max_retries,
                    ablation_components=ablation_components,
                    ablation_suffix=ablation_suffix,
                    thinking=thinking,
                    domains_config=domains_config,
                    example_variant=args.example,
                    hints_variant=args.hints,
                )

                if success:
                    total_success += 1
                    if usage:
                        cumulative_usage['input_tokens'] += usage['input_tokens']
                        cumulative_usage['output_tokens'] += usage['output_tokens']
                        cumulative_usage['thinking_tokens'] += usage.get('thinking_tokens', 0)
                        cumulative_usage['cost'] += usage['cost']
                        cumulative_usage['api_calls'] += 1
                else:
                    total_failed += 1

                if args.dry_run:
                    break

                if args.delay > 0 and i < args.start_index + batch_count - 1:
                    time.sleep(args.delay)

        else:
            # --- Iterative mode: --count N iterations of refinement ---
            iterations = args.count
            if iterations <= 1:
                # Single-shot: same as old behavior for count=1
                print(f"Mode: Single-shot (1 heuristic, no refinement)")
                print(f"{'='*60}")

                success, usage = generate_for_domain(
                    domain_key=domain_key,
                    domain_config=domain_config,
                    model_config=model_config,
                    defaults=defaults,
                    counter=args.start_index,
                    api_key=api_key,
                    dry_run=args.dry_run,
                    max_retries=args.max_retries,
                    ablation_components=ablation_components,
                    ablation_suffix=ablation_suffix,
                    thinking=thinking,
                    domains_config=domains_config,
                    example_variant=args.example,
                    hints_variant=args.hints,
                )
                if success:
                    total_success += 1
                    if usage:
                        cumulative_usage['input_tokens'] += usage['input_tokens']
                        cumulative_usage['output_tokens'] += usage['output_tokens']
                        cumulative_usage['thinking_tokens'] += usage.get('thinking_tokens', 0)
                        cumulative_usage['cost'] += usage['cost']
                        cumulative_usage['api_calls'] += 1
                else:
                    total_failed += 1
            else:
                # Iterative refinement
                print(f"Mode: Iterative refinement ({iterations} iterations)")
                print(f"{'='*60}")

                if args.dry_run:
                    # In dry-run, just show what would happen
                    naming = get_naming_versioned(domain_config['name'], model_config['name'], args.start_index, 0,
                                                  domain_config.get('file_prefix'), ablation_suffix=ablation_suffix)
                    prompt = build_base_prompt(domain_config, model_config, defaults, naming, args.start_index, ablation_components,
                                               domain_key, domains_config, args.example, args.hints)
                    print(f"\nDRY RUN: Would generate {iterations} iterations for index {args.start_index}")
                    print(f"  v0: {naming['filename']}")
                    for v in range(1, iterations):
                        vn = get_naming_versioned(domain_config['name'], model_config['name'], args.start_index, v,
                                                  domain_config.get('file_prefix'), ablation_suffix=ablation_suffix)
                        print(f"  v{v}: {vn['filename']}")
                    print(f"\nBase prompt length: {len(prompt)} characters")
                    print(f"\nPrompt preview (first 2000 chars):")
                    print("-" * 40)
                    print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
                    print("-" * 40)
                else:
                    result = generate_iterative(
                        domain_key=domain_key,
                        domain_config=domain_config,
                        model_config=model_config,
                        defaults=defaults,
                        counter=args.start_index,
                        iterations=iterations,
                        api_key=api_key,
                        max_retries=args.max_retries,
                        delay=args.delay,
                        ablation_components=ablation_components,
                        ablation_suffix=ablation_suffix,
                        thinking=thinking,
                        domains_config=domains_config,
                        example_variant=args.example,
                        hints_variant=args.hints,
                        )
                    total_success += len(result['versions'])
                    iter_usage = result.get('total_usage', {})
                    cumulative_usage['input_tokens'] += iter_usage.get('input_tokens', 0)
                    cumulative_usage['output_tokens'] += iter_usage.get('output_tokens', 0)
                    cumulative_usage['thinking_tokens'] += iter_usage.get('thinking_tokens', 0)
                    cumulative_usage['cost'] += iter_usage.get('cost', 0.0)
                    cumulative_usage['api_calls'] += iter_usage.get('api_calls', 0)

                    # Print summary of all versions
                    print(f"\n  --- Iteration Summary ---")
                    for v_info in result['versions']:
                        v = v_info['version']
                        tr = v_info.get('test_results')
                        status = tr.get('status', 'not tested') if tr else 'not tested'
                        nodes = tr.get('nodes_expanded', 'N/A') if tr else 'N/A'
                        print(f"    v{v}: {v_info['file_path']} (status={status}, nodes={nodes})")

    # Summary
    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"  Successful: {total_success}")
        print(f"  Failed: {total_failed}")
        if cumulative_usage['api_calls'] > 0:
            print(f"\nCost summary: {cumulative_usage['api_calls']} API calls, "
                  f"{cumulative_usage['input_tokens']:,} input / "
                  f"{cumulative_usage['output_tokens']:,} output / "
                  f"{cumulative_usage['thinking_tokens']:,} reasoning tokens, "
                  f"Total: ${cumulative_usage['cost']:.4f}")
        print(f"{'='*60}")

    # Exit non-zero when anything failed, so a driver can see it. Without this a
    # sweep where every call 404s still reported "OK" per tuple and kept going.
    if total_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
