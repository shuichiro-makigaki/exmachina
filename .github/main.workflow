workflow "pytest" {
  on = "push"
  resolves = ["docker://python:3."]
}

action "pip install" {
  uses = "docker://python:3.6"
  args = "pip install --user -r requirements.txt"
}

action "docker://python:3." {
  uses = "docker://python:3.6"
  needs = ["pip install"]
  args = "pytest -v --cov . --pep8 --junitxml=test_reports/junit.xml"
}
