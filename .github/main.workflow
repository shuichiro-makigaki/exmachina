workflow "pytest" {
  on = "push"
  resolves = [
    "Filters for GitHub Actions",
  ]
}

action "Filters for GitHub Actions" {
  uses = "actions/bin/sh@master"
  args = "export && pwd && ls -l"
}
