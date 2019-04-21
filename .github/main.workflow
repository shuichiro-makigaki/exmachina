workflow "main" {
  on = "push"
  resolves = ["pytest"]
}

action "pip install" {
  uses = "docker://python:3.6"
  args = "pip install --user -r requirements.txt"
}

action "pytest" {
  uses = "docker://python:3.6"
  needs = ["pip install"]
  args = "/github/home/.local/bin/pytest -v --cov . --pep8 --junitxml=test_reports/junit.xml"
}
