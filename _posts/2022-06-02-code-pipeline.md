---
title: "Setting up AWS Code Pipeline to automate deployment of tweets streaming application"
categories: ['serverless','lambda','code pipeline', 'ci-cd']
tags: ['serverless','lambda','code pipeline', 'ci-cd']
url: https://github.com/ryankarlos/AWS-CICD
comments: true
---
___

{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-CICD/master/projects/deploy-lambda-image/README.md %}
{% endcapture %}



{{ products | replace_first:"## AWS CodePipeline" }}
