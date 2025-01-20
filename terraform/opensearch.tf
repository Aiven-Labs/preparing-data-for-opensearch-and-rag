variable "aiven_os_service_name" {
  type = string
}

resource "aiven_opensearch" "os1" {
  project      = data.aiven_project.langchain_demo.project
  cloud_name   = "google-us-east1"
  plan         = "startup-4"
  service_name = var.aiven_os_service_name

}

output "os_service_endpoint_uri" {
  value     = aiven_opensearch.os1.service_uri
  sensitive = true
}

output "os_service_dashboard" {
  value     = aiven_opensearch.os1.service_host
  sensitive = true
}

output "os_password" {
  value     = aiven_opensearch.os1.service_password
  sensitive = true
}


