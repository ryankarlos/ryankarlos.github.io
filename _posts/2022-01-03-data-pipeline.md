---
title: "Using AWS Data Pipeline to automate movement of data between AWS services"
categories: ['serverless','data-pipeline','s3', 'RDS', 'Redshift']
tags: ['serverless','data-pipeline','s3', 'RDS', 'Redshift']
url: https://github.com/ryankarlos/AWS-ETL-Workflows
comments: true
---
___

AWS Data Pipeline is a service that helps automate movement of data between different AWS compute and storage services such as Amazon S3, Amazon RDS, Amazon DynamoDB, and Amazon EMR.
This helps create complex data processing workloads that are fault tolerant, repeatable, and highly available.

{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-ETL-Workflows/master/data-pipelines/README.md %}
{% endcapture %}

{% capture products1 %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-ETL-Workflows/master/data-pipelines/s3_to_redshift/README.md %}
{% endcapture %}


{{ products | replace_first:"## Creating the different DataPipelines" | replace_first:"Refer to these [instructions](s3_to_redshift)" }}


{{products1 | replace_first:'# Data Pipeline S3 to Redshift' | replace: '##', "#####"}}