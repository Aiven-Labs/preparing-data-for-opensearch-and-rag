variable "aiven_project_name" {
  type = string
}

terraform {
  required_providers {
    aiven = {
      source  = "aiven/aiven"
      version = ">= 4.0.0, < 5.0.0"
    }
  }
}

data "aiven_project" "langchain_demo" {
  project = var.aiven_project_name
}
