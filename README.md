# LLM-Powered Document Query System

## Overview
This project implements an intelligent document querying system using LlamaIndex and OpenAI's GPT models. It creates a hierarchical agent system that can analyze and respond to queries across multiple document sources with specialized knowledge domains.

## Features
- Multi-document source handling
- Hierarchical agent architecture
- Vector and summary-based indexing
- Persistent storage of vector indices
- Specialized agents for different knowledge domains
- Top-level agent for coordinating responses

## Prerequisites
- Python 3.x
- OpenAI API key
- LlamaIndex library
- Required document sources in text format

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install llama-index openai
```
3. Set up your OpenAI API key in the environment variables or in the code

## Project Structure
```
project/
├── data/                      # Document source files
│   ├── root_cause_analysis.txt
│   ├── product_interview_framework.txt
│   ├── pm_interview.txt
│   └── planning_mvp.txt
├── data_source/              # Persistent vector indices
├── main.py                   # Main application code
└── README.md
```

## Configuration
The system is configured to work with the following data sources:
- Root Cause Analysis
- Product Interview Framework
- PM Interview
- Planning MVP

## Usage
1. Place your text documents in the `data/` directory
2. Run the main script:
```bash
python main.py
```

## How It Works
1. **Document Processing**: The system reads and processes documents using LlamaIndex's SimpleDirectoryReader.

2. **Index Creation**: For each data source, the system creates:
   - Vector Store Index for specific queries
   - Summary Index for holistic information
   - Persistent storage for vector indices

3. **Agent Hierarchy**:
   - Individual agents for each knowledge domain
   - Top-level agent for coordinating responses
   - Base query engine for fallback responses

## Key Components
1. **Data Source Management**
```python:main.py
startLine: 34
endLine: 42
```

2. **Query Engine and Agent Creation**
```python:main.py
startLine: 44
endLine: 113
```

3. **Top-Level Agent**
```python:main.py
startLine: 140
endLine: 151
```

## Example Query
```python
query = "how do you perform root cause analysis to understand reduction in users pressing the add to cart bar button?"
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[Add your license information here]

## Contact
[Add your contact information here]
