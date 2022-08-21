---
layout: post
title: Deploying a Flask App to EC2 with RDS and Redshift
categories: ['ec2','code deploy', 'webserver' , 'rds' , 'redshift']
tags: ['flask','ec2' , 'code deploy' , 'webserver' , 'rds' , 'redshift']
url: https://github.com/ryankarlos/AWS-VPC/tree/master/aws_vpc
comments: true
---
___

{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-VPC/master/aws_vpc/README.md %}
{% endcapture %}

{% assign title = page.title | prepend: "## "   %}

{{ products | remove_first: title | remove: '<a href="https://ryankarlos.github.io/AWS-VPC/">Home</a>'}}