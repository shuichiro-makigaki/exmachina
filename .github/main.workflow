workflow "pytest" {
  on = "push"
  resolves = ["docker://python:3.6"]
}

action "Filters for GitHub Actions" {
  uses = "docker://python:3.6"
  args = "pip install --user -r requirements.txt"
}

action "docker://python:3.6" {
  uses = "docker://python:3.6"
  needs = ["Filters for GitHub Actions"]
  args = "pip list"
}
