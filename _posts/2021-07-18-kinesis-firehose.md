---
title: "Streaming tweets using AWS kinesis data streams and firehose"
categories: ['streaming','kinesis', 'firehose', 'tweets']
tags: ['streaming','kinesis', 'firehose', 'tweets']
url: https://github.com/ryankarlos/AWS-ETL-Workflows
comments: true
---
___
{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-ETL-Workflows/master/kinesis/README.md %}
{% endcapture %}

{% assign title="# Streaming tweets using AWS kinesis data streams and firehose" %}

{{ products | replace_first:title  }}
