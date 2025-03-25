# Contributing to G-FOLD

Thank you for your interest in contributing to G-FOLD! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed(these are recommended versions):
- Python 3.12
- [Rust](https://www.rust-lang.org/tools/install)
- [Eigen](https://github.com/oxfordcontrol/Clarabel.cpp#installation)
- Tkinter (`sudo apt-get install python3.12-tk`)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/g-fold.git
   cd g-fold/generator
   ```
3. Install the package in development mode(if needed):
   ```bash
   pip install -e .
   ```

## Making Contributions

### Branch Names

- Use descriptive branch names that reflect the change you're making
- Format: `type/description`, e.g., `feature/add-new-solver` or `fix/memory-leak`

### Pull Requests

1. Create a new branch for your changes
2. Make your changes
3. Push to your fork
4. Open a Pull Request with a clear description of the changes
5. Wait for review

## Development Guidelines

### Documentation

- Update the README.md if you change functionality
- Add docstrings to new functions and classes
- Include examples for new features

### Testing

Testing infrastructure will be added in the future. Once available, new features should include appropriate tests.

## Reporting Issues

### Bug Reports

When reporting bugs, please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and operating system
- Any relevant error messages

### Feature Requests

For feature requests, please:
- Clearly describe the feature
- Explain the use case
- Provide examples of how it would be used

---

Thank you for contributing to G-FOLD! Your efforts help make this project better for everyone. 