workflow "pytest" {
  on = "push"
  resolves = ["docker://python:3.6-1", "docker://python:3.6"]
}

action "docker://python:3.6" {
  uses = "docker://python:3.6"
  args = "pip install -r requirements.txt"
}

action "docker://python:3.6-1" {
  uses = "docker://python:3.6"
  needs = ["docker://python:3.6"]
  args = "pytest -v --cov . --pep8 --junitxml=test_reports/junit.xml"
}
