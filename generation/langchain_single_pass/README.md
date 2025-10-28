# NKI Kernel Generation Pipeline

This directory contains the refactored NKI kernel generation pipeline, organized into a clean, modular structure.

## Directory Structure

```
generation/langchain_single_pass/
├── core/                           # Core configuration and management
│   ├── __init__.py
│   ├── config.py                   # Configuration and paths
│   ├── llm_manager.py              # LLM initialization and management
│   └── logging_manager.py          # Centralized logging
├── pipeline/                       # Main pipeline components
│   ├── __init__.py
│   ├── kernel_generator.py         # Main generation logic
│   ├── error_handler.py            # Error parsing and documentation
│   ├── test_runner.py              # Test execution and validation
│   └── iteration_manager.py        # Iterative improvement loop
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── file_utils.py               # File operations
│   ├── prompt_utils.py             # Prompt generation and management
│   └── json_utils.py               # JSON parsing utilities
├── operators/                      # Operator-specific logic
│   ├── __init__.py
│   ├── operator_config.py          # Operator definitions and categories
│   └── operator_runner.py          # Operator-specific execution
├── main.py                         # Main entry point
└── README.md                       # This file
```

## Key Components

### Core Module
- **config.py**: Centralized configuration management with configurable paths, LLM settings, and retry logic
- **llm_manager.py**: Handles direct Bedrock API calls and retry logic
- **logging_manager.py**: Centralized logging with consolidated iteration tracking

### Pipeline Module
- **kernel_generator.py**: Main orchestrator for the kernel generation process
- **error_handler.py**: Manages error parsing, documentation loading, and change analysis
- **test_runner.py**: Handles test execution using the tests.py module
- **iteration_manager.py**: Manages the iterative improvement loop

### Utils Module
- **file_utils.py**: File reading, writing, and logging utilities
- **prompt_utils.py**: Prompt generation and kernel code extraction
- **json_utils.py**: JSON parsing and extraction utilities

### Operators Module
- **operator_runner.py**: Handles operator execution with retry logic
- **operator_config.py**: Defines operator categories and test mappings

## Usage

### Running the Pipeline

To run the pipeline, execute the main entry point:

```bash
cd generation/langchain_single_pass
python main.py
```

### Configuration

The pipeline is configured through the `core/config.py` file. Key configuration options:

- **Base paths**: Configurable for different environments
- **LLM settings**: Model ID, temperature, retry settings
- **Retry logic**: Configurable retry attempts and backoff
- **Operator categories**: Elementwise, multi-element, and product operators

### Adding New Operators

To add new operators:

1. Add the operator name to the appropriate category in `core/config.py`
2. Add the corresponding test function name
3. Create a prompt file in the `prompts/` directory
4. Add a test function to `tests.py`

## Key Improvements

1. **Modular Design**: Clean separation of concerns with dedicated modules
2. **Configurable**: All paths and settings are configurable
3. **Maintainable**: Each component has a single responsibility
4. **Testable**: Individual components can be tested in isolation
5. **Extensible**: Easy to add new operators or modify behavior

## Dependencies

The pipeline maintains the same dependencies as the original:
- `extraction.py`: Kernel extraction and test execution
- `doc_grabber.py`: Documentation loading and selection
- `nki_error_parsing.py`: Error parsing and documentation
- `tests.py`: Test functions for validation

## Migration from Original

The refactored pipeline maintains the same functionality as `all_in_one_generator_new.py` but with:

- Better organization and readability
- Configurable settings
- Modular architecture
- Improved maintainability
- Same error handling and logging capabilities 