workflow "pytest" {
  on = "push"
  resolves = ["actions/bin/sh@master"]
}

action "Filters for GitHub Actions" {
  uses = "docker://python:3.6"
  args = "pip install --user -r requirements.txt"
}

action "actions/bin/sh@master" {
  uses = "actions/bin/sh@master"
  needs = ["Filters for GitHub Actions"]
  args = ["ls -al", "pwd", "export"]
}
