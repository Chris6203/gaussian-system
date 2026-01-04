# Contributing to Quantor-MTFuzz

Welcome! This document outlines the process for team developers to contribute to the **Quantor-MTFuzz** project safely and professionally.

---

## 🔐 Repository Access

This is a private repository. To contribute, you must be invited as a **Collaborator** by the repository owner.

- **Current Status**: See [DEVELOPMENT_STATUS.md](file:///c:/SPYOptionTrader_test/DEVELOPMENT_STATUS.md) for a list of resolved bugs and the current roadmap.
- **Authentication**: Do **not** use the owner's credentials. Use your own GitHub account.
- **Tokens**: When performing Git operations over HTTPS, use a **Personal Access Token (PAT)** instead of your password. 
  - *Setup*: GitHub Settings → Developer Settings → Personal Access Tokens → Tokens (classic).

---

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/trextrader/spy-iron-condor-trading.git
   cd spy-iron-condor-trading
   ```

2. **Environment Setup**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Local Configuration**:
   Create your own `core/config.py` from the template:
   ```powershell
   copy core/config.template.py core/config.py
   ```
   *Note: `core/config.py` is git-ignored and will not be pushed.*

---

## 🛠️ Development Workflow

1. **Feature Branching**: Create a new branch for every feature or bugfix.
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit Standard**: Use descriptive commit messages.
   ```bash
   git commit -m "Refactor: Optimized leg-sync logic in backtest_engine"
   ```
3. **Synchronization**: Keep your local branch updated with the main branch.
   ```bash
   git pull origin main
   ```
4. **Pull Requests**: Push your branch to GitHub and open a Pull Request (PR) for review.

---

## 📐 Coding Standards

- **Formatting**: Follow PEP 8 for Python code.
- **Documentation**: Update the `README.md` if you add new command-line arguments or modify core mathematical foundations.
- **Testing**: Ensure all backtests run successfully before submitting a PR.

---

## ⚖️ Security Notice
Never commit API keys or secrets to the repository. Ensure all sensitive information is kept in `core/config.py` or passed via environment variables.
